# 导入必要库
import os
import re
import random
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from transformers.optimization import get_cosine_schedule_with_warmup
from rank_bm25 import BM25Okapi
import numpy as np
import torch.nn.functional as F

# 环境配置
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
torch.backends.cudnn.benchmark = True

##################################################
#                  超参数配置                   #
##################################################
CONFIG = {
    'model_name': 'hfl/chinese-roberta-wwm-ext',
    'max_length': 32,
    'projection_dim': 256,
    'margin': 0.8,
    'batch_size': 16,
    'gradient_accumulation': 4,
    'learning_rate': 5e-5,
    'weight_decay': 0.001,
    'epochs': 20,
    'patience': 5,
    'dropout': 0.2,
    'train_path': 'xlj.xlsx',
    'test_path': 'csj.xlsx',
    'save_path': 'clinical_roberta.pth',
    'ce_weight': 0.7,
    'hf_hub_disable_symlinks': 'True'
}

# 医学术语同义词表
MEDICAL_SYNONYMS = {
    '糖尿病': ['DM', '消渴症'],
    '高血压': ['HTN', '原发性高血压'],
    '冠心病': ['冠状动脉粥样硬化性心脏病'],
    '肺炎': ['肺部感染']
}


##################################################
#                  数据增强模块                 #
##################################################
class MedicalAugmenter:
    """医疗文本数据增强器"""

    @classmethod
    def augment(cls, text):
        # 统一处理空值
        if not isinstance(text, str) or len(text) == 0:
            return text

        # 同义词替换（50%概率）
        for term, syns in MEDICAL_SYNONYMS.items():
            if term in text and random.random() < 0.5:
                text = text.replace(term, random.choice(syns))

        # 数字扰动（30%概率）
        if random.random() < 0.3:
            text = re.sub(r'\d+', lambda m: str(int(m.group()) + random.randint(0, 2)), text)

        return text.strip()


##################################################
#                  数据加载器                   #
##################################################
class ClinicalDataset(Dataset):
    """医疗术语数据集加载器"""

    def __init__(self, df, tokenizer, is_train=False):
        self.df = self._clean_data(df)
        self.tokenizer = tokenizer
        self.is_train = is_train

        # 构建标签映射
        self.standard_list = list(self.df['standard'].unique())
        self.label_map = {term: idx for idx, term in enumerate(self.standard_list)}
        self.num_classes = len(self.standard_list)

        # 训练时构建BM25检索库
        if is_train:
            self._build_bm25_corpus()

    def _build_bm25_corpus(self):
        """构建BM25检索库"""
        self.corpus = list(set(self.df['original']) | set(self.df['standard']))
        self.bm25 = BM25Okapi([doc.split('-') for doc in self.corpus])

    def _clean_data(self, df):
        """数据清洗"""
        # 列名标准化
        df = df.rename(columns={
            '原始诊断名称': 'original',
            '映射结果中文同义词': 'standard'
        }).dropna()

        # 字符标准化
        char_map = str.maketrans('１２３４５６７８９０', '1234567890')
        df['original'] = df['original'].str.translate(char_map)
        df['standard'] = df['standard'].str.translate(char_map)

        # 过滤无效数据
        df = df[
            (df['original'] != df['standard']) &
            (df['original'].str.len() >= 2) &
            (df['standard'].str.len() >= 2)
            ]
        return df.drop_duplicates(subset=['original'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        anchor = row['original']
        positive = row['standard']
        label = self.label_map[positive]

        # 训练时数据增强
        if self.is_train:
            anchor = MedicalAugmenter.augment(anchor)
            positive = MedicalAugmenter.augment(positive)
            negative = self._get_hard_negative(anchor)
        else:
            negative = None

        return {
            'anchor': self._encode(anchor),
            'positive': self._encode(positive),
            'negative': self._encode(negative) if negative else None,
            'original_text': row['original'],
            'standard_text': row['standard'],
            'label': label
        }

    def _encode(self, text):
        """安全编码函数"""
        if text is None:
            return None
        return self.tokenizer(
            text,
            max_length=CONFIG['max_length'],
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_token_type_ids=False  # 明确禁用token_type_ids
        )

    def _get_hard_negative(self, query):
        """获取困难负样本"""
        tokenized_query = query.split('-')
        scores = self.bm25.get_scores(tokenized_query)
        top_idx = np.argmax(scores)
        return self.corpus[top_idx] if self.corpus[top_idx] != query else None


##################################################
#                  模型架构                    #
##################################################
class ClinicalRoberta(nn.Module):
    """修复版医疗术语标准化模型"""

    def __init__(self, num_classes):
        super().__init__()
        # 主干网络
        self.encoder = AutoModel.from_pretrained(
            CONFIG['model_name'],
            hidden_dropout_prob=CONFIG['dropout'],
            attention_probs_dropout_prob=CONFIG['dropout'],
            add_pooling_layer=False  # 避免不必要的池化层
        )

        # 冻结前3层
        for layer in self.encoder.encoder.layer[:3]:
            for param in layer.parameters():
                param.requires_grad = False

        # 投影头
        self.projection = nn.Sequential(
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Dropout(CONFIG['dropout']),
            nn.LayerNorm(512),
            nn.Linear(512, CONFIG['projection_dim']),
            nn.LayerNorm(CONFIG['projection_dim'])
        )

        # 分类器
        self.classifier = nn.Linear(CONFIG['projection_dim'], num_classes)

    def forward(self, input_ids, attention_mask):
        """修复前向传播参数"""
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        # 使用首token作为特征
        pooled = outputs.last_hidden_state[:, 0]
        projected = self.projection(pooled)
        logits = self.classifier(projected)
        return projected, logits


##################################################
#                  训练引擎                    #
##################################################
class TripletTrainer:
    """修复版训练引擎"""

    def __init__(self, model, train_loader, test_loader):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader

        # 优化器
        self.optimizer = optim.AdamW([
            {'params': model.encoder.parameters(), 'lr': 1e-5},
            {'params': model.projection.parameters(), 'lr': 3e-5},
            {'params': model.classifier.parameters(), 'lr': 5e-5}
        ], weight_decay=CONFIG['weight_decay'])

        # 学习率调度
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=100,
            num_training_steps=len(train_loader) * CONFIG['epochs']
        )

        # 损失函数
        self.triplet_loss = nn.TripletMarginLoss(margin=CONFIG['margin'])
        self.ce_loss = nn.CrossEntropyLoss()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.device.type == 'cuda')  # 自动检测CUDA

        # 训练状态
        self.best_acc = 0.0
        self.no_improve = 0

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0

        with tqdm(self.train_loader, desc=f"Epoch {epoch}") as pbar:
            for step, batch in enumerate(pbar):
                self.optimizer.zero_grad()

                # 准备输入
                anchor = self._prepare_input(batch['anchor'])
                positive = self._prepare_input(batch['positive'])
                negative = self._prepare_input(batch['negative'])
                labels = batch['label'].to(self.device)

                # 混合精度训练
                with torch.autocast(device_type=self.device.type, enabled=self.device.type == 'cuda'):
                    anchor_emb, anchor_logits = self.model(**anchor)
                    positive_emb, _ = self.model(**positive)
                    negative_emb = self._get_neg_emb(positive_emb, negative)

                    # 计算损失
                    triplet_loss = self.triplet_loss(anchor_emb, positive_emb, negative_emb)
                    ce_loss = self.ce_loss(anchor_logits, labels)
                    loss = triplet_loss + CONFIG['ce_weight'] * ce_loss

                # 反向传播
                self.scaler.scale(loss).backward()

                # 梯度裁剪与更新
                if (step + 1) % CONFIG['gradient_accumulation'] == 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                # 更新进度
                total_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=self.scheduler.get_last_lr()[0])

        return total_loss / len(self.train_loader)

    def _get_neg_emb(self, positive_emb, negative):
        """安全获取负样本嵌入"""
        if negative is not None:
            return self.model(**negative)[0]
        # 随机负样本
        return positive_emb[torch.randperm(positive_emb.size(0))]

    def _prepare_input(self, inputs):
        """输入预处理"""
        if inputs is None:
            return None
        return {
            'input_ids': inputs['input_ids'].squeeze(1).to(self.device),
            'attention_mask': inputs['attention_mask'].squeeze(1).to(self.device)
        }

    def evaluate(self):
        """修复版评估方法"""
        self.model.eval()
        standards = list(set([term for batch in self.test_loader for term in batch['standard_text']]))
        standard_embs = []

        # 预计算标准嵌入
        with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=self.device.type == 'cuda'):
            for term in standards:
                encoded = self.test_loader.dataset.tokenizer(
                    term,
                    max_length=CONFIG['max_length'],
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt',
                    return_token_type_ids=False  # 确保没有token_type_ids
                ).to(self.device)
                emb, _ = self.model(
                    input_ids=encoded['input_ids'],
                    attention_mask=encoded['attention_mask']
                )
                standard_embs.append(F.normalize(emb, p=2, dim=1))
        standard_embs = torch.cat(standard_embs, dim=0)

        # 评估
        correct = 0
        total = 0
        for batch in tqdm(self.test_loader, desc="评估中"):
            anchor = self._prepare_input(batch['anchor'])

            with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=self.device.type == 'cuda'):
                anchor_emb, _ = self.model(**anchor)
                anchor_emb = F.normalize(anchor_emb, p=2, dim=1)

                # 相似度计算
                similarities = torch.mm(anchor_emb, standard_embs.T)
                _, top_preds = torch.max(similarities, dim=1)

            # 统计正确率
            for idx, pred in enumerate(top_preds):
                if standards[pred.item()] == batch['standard_text'][idx]:
                    correct += 1
                total += 1

        return correct / total


##################################################
#              预测结果输出模块                 #
##################################################
def print_predictions(model, test_loader):
    """修复版预测输出"""
    device = next(model.parameters()).device
    model.eval()

    # 构建标准库
    standards = list(set([term for batch in test_loader for term in batch['standard_text']]))
    standard_embs = []

    # 预计算嵌入
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
        for term in standards:
            encoded = test_loader.dataset.tokenizer(
                term,
                max_length=CONFIG['max_length'],
                padding='max_length',
                truncation=True,
                return_tensors='pt',
                return_token_type_ids=False
            ).to(device)
            emb, _ = model(
                input_ids=encoded['input_ids'],
                attention_mask=encoded['attention_mask']
            )
            standard_embs.append(F.normalize(emb, p=2, dim=1))
    standard_embs = torch.cat(standard_embs, dim=0)

    # 收集结果
    results = []
    for batch in tqdm(test_loader, desc="生成预测"):
        anchor = {
            'input_ids': batch['anchor']['input_ids'].squeeze(1).to(device),
            'attention_mask': batch['anchor']['attention_mask'].squeeze(1).to(device)
        }

        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            anchor_emb, _ = model(**anchor)
            anchor_emb = F.normalize(anchor_emb, p=2, dim=1)

            # 匹配最相似标准
            similarities = torch.mm(anchor_emb, standard_embs.T)
            _, top_indices = torch.max(similarities, dim=1)

        # 记录结果
        for i in range(len(top_indices)):
            orig = batch['original_text'][i]
            pred = standards[top_indices[i].item()]
            true = batch['standard_text'][i]
            results.append((orig, pred, true))

    # 打印结果
    print("\n{:<30} {:<30} {:<30} {}".format("原始术语", "预测术语", "真实答案", "状态"))
    print("=" * 95)
    correct = 0
    for orig, pred, true in results[:20]:
        status = "✓" if pred == true else "✗"
        print("{:<30} {:<30} {:<30} {}".format(orig, pred, true, status))
        if pred == true: correct += 1

    # 统计信息
    total = len(results)
    accuracy = correct / total
    print("\n" + "=" * 95)
    print(f"总样本数: {total} | 正确数: {correct} | 准确率: {accuracy * 100:.2f}%")


##################################################
#                  主流程                      #
##################################################
def medical_collate(batch):
    """批次处理函数"""
    processed = {
        'anchor': {
            'input_ids': torch.stack([x['anchor']['input_ids'][0] for x in batch]),
            'attention_mask': torch.stack([x['anchor']['attention_mask'][0] for x in batch])
        },
        'positive': {
            'input_ids': torch.stack([x['positive']['input_ids'][0] for x in batch]),
            'attention_mask': torch.stack([x['positive']['attention_mask'][0] for x in batch])
        },
        'original_text': [x['original_text'] for x in batch],
        'standard_text': [x['standard_text'] for x in batch],
        'label': torch.tensor([x['label'] for x in batch], dtype=torch.long)
    }
    # 处理负样本
    negatives = [x['negative'] for x in batch if x['negative']]
    if len(negatives) == len(batch):
        processed['negative'] = {
            'input_ids': torch.stack([x['negative']['input_ids'][0] for x in batch]),
            'attention_mask': torch.stack([x['negative']['attention_mask'][0] for x in batch])
        }
    else:
        processed['negative'] = None
    return processed


if __name__ == "__main__":
    # 配置镜像源
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

    # 数据加载
    print("加载数据...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    train_df = pd.read_excel(CONFIG['train_path'])
    test_df = pd.read_excel(CONFIG['test_path'])

    # 数据预处理
    print("预处理数据...")
    train_set = ClinicalDataset(train_df, tokenizer, is_train=True)
    test_set = ClinicalDataset(test_df, tokenizer)

    # 模型初始化
    print("初始化模型...")
    model = ClinicalRoberta(num_classes=train_set.num_classes)

    # 数据加载器
    test_loader = DataLoader(
        test_set,
        batch_size=CONFIG['batch_size'] * 2,
        collate_fn=medical_collate,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # 训练初始化
    print("初始化训练器...")
    trainer = TripletTrainer(
        model,
        DataLoader(
            train_set,
            batch_size=CONFIG['batch_size'],
            shuffle=True,
            collate_fn=medical_collate,
            num_workers=0,
            pin_memory=True
        ),
        test_loader
    )

    # 训练循环
    print("开始训练...")
    best_acc = 0.0
    for epoch in range(1, CONFIG['epochs'] + 1):
        train_loss = trainer.train_epoch(epoch)
        test_acc = trainer.evaluate()

        print(f"Epoch {epoch} | 训练损失: {train_loss:.4f} | 验证准确率: {test_acc:.4f}")

        # 模型保存与早停
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), CONFIG['save_path'])
            print(f"发现最佳模型，准确率 {test_acc:.4f}，模型已保存")
            trainer.no_improve = 0
        else:
            trainer.no_improve += 1
            if trainer.no_improve >= CONFIG['patience']:
                print(f"早停触发，停止训练")
                break

    # 最终预测
    print("\n加载最佳模型生成预测...")
    model.load_state_dict(torch.load(CONFIG['save_path']))
    print_predictions(model, test_loader)