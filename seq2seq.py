import torch
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import os
import pandas as pd
import re
import torchtext


# 常量
_WORD_SPLIT = re.compile(r"([.,!?\"':;)(])")     # 正则表达式模式
TRAIN_FNAME = "e2e_dataset/trainset.csv"         # 训练集路径 
DEV_FNAME = "e2e_dataset/devset.csv"             # 验证集路径
TEST_FNAME = "e2e_dataset/testset.csv"           # 测试集路径
DATA_LEN_HIST_PATH = "result/data_len_hist.png"  # 句长直方图存储路径
SRC_VOCAB = []                                   # 输入词汇词典
TGT_VOCAB = []                                   # 输出词汇词典
MAX_LEN = 50                                     # 最大句长
SOS_TOKEN = '<sos>'                              # 句子起始标记
EOS_TOKEN = '<eos>'                              # 句子结束标记


'''Config类用于设置实验参数'''
class Config:
    def __init__(self):
        self.save_data_path = "result/result.txt"     # 结果存储路径
        self.encoder_save_path = "model/encoder.mdl"  # 编码器模型存储路径
        self.decoder_save_path = "model/decoder.mdl"  # 解码器模型存储路径
        self.plot_save_path = "result/loss_plot.png"  # 训练集损失图存储路径
        self.input_dim = len(SRC_VOCAB)               # 编码器输入维度
        self.output_dim = len(TGT_VOCAB)              # 解码器输出维度
        self.hidden_size = 256                        # 隐藏层维度
        self.device = "cpu"                          # 设备
        self.lr = 0.01                                # 学习率
        self.max_len = MAX_LEN                        # 最大句长
        self.batch_size = 1                           # 批次大小 
        self.max_epoch = 1                            # 最大迭代次数


'''数据分析'''
class Data_analysis():
    def __init__(self, save_path=DATA_LEN_HIST_PATH):
        data = pd.read_csv(TRAIN_FNAME,dtype=str) 
        references = [self.process_e2e_text(data.iloc[i]['ref']) for i in range(len(data))]
        self.references_lens = [len(d) for d in references]
        self.plot_len_hist(self.references_lens, save_path)

    def process_e2e_text(self,s):   # 将每个ref表示为token列表。
        words = []
        for fragment in s.strip().split():
            fragment_tokens = _WORD_SPLIT.split(fragment)
            words.extend(fragment_tokens)
        tokens = [w for w in words if w]
        return tokens

    def cnt_bins_and_cnts(self):   # 用于计算句子长度的确切数量
        lengths_to_consider = [0,10,20,30,40,50,60,70,80]
        bins = [(lengths_to_consider[i], lengths_to_consider[i+1]) \
                for i in range(len(lengths_to_consider)-1)]
        cnts = [0] * len(bins)
        for l in self.references_lens:
            for bin_idx, b in enumerate(bins):
                if l > b[0] and l <= b[1]:
                    cnts[bin_idx] += 1
                    break
        return (bins, cnts)

    def plot_len_hist(self, lens, fname):   # 绘制参考长度分布直方图
        references_lens_df = pd.DataFrame(self.references_lens)
        mean = float(references_lens_df.mean())   # 检索统计数据
        std = float (references_lens_df.std())
        min_len = int(references_lens_df.min())
        max_len = int(references_lens_df.max())

        # 绘制长度分布的直方图
        plt.figure(0)    
        plt.rcParams.update({'font.size': 12})
        n, bins, patches = plt.hist(lens, 20, facecolor='b', alpha=0.55)
        plt.xlabel('Sentence Length')
        plt.ylabel('Number of sentences')
        plt.title('Sentence length distribution')
        plt.axis([0, 80, 0, 10000])
        plt.text(40, 7500, r'$mean={:.2f},\ std={:.2f}$'.format(mean, std))
        plt.text(40, 6800, r'$min={},\ max={}$'.format(min_len, max_len))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(fname)
        plt.show()


'''读取数据'''
class E2EDataset(Dataset):
    def __init__(self):
        '''
            __init__()函数的内容：
            1.读取数据集
            2.对读取的数据表做序列化处理
            3.构建输入(序列化后的数据表)和输出(参考文本)的词典
        '''
        src = torchtext.data.Field(
            batch_first=True, 
            include_lengths=True
            )
        tgt = torchtext.data.Field(
            batch_first=True, 
            preprocessing = lambda seq: [SOS_TOKEN] + seq + [EOS_TOKEN]
            )
        self.train_data = torchtext.data.TabularDataset(
                path=TRAIN_FNAME, format='csv',
                fields=[('src', src), ('tgt', tgt)],
                filter_pred=self.len_filter
            )
        self.dev_data = torchtext.data.TabularDataset(
                path=DEV_FNAME, format='csv',
                fields=[('src', src), ('tgt', tgt)],
                filter_pred=self.len_filter
            )
        self.test_data = torchtext.data.TabularDataset(
                path=TEST_FNAME, format='csv',
                fields=[('src', src)],
                filter_pred=self.len_filter_test
            )
        src.build_vocab(self.train_data.src, self.dev_data.src, self.test_data.src, 
                        max_size=50000)
        tgt.build_vocab(self.train_data.tgt, self.dev_data.tgt, self.test_data.tgt, 
                        max_size=50000)
        self.src_vocab = src.vocab
        self.tgt_vocab = tgt.vocab
        self.print_dataset_info(self.train_data, src, tgt, self.src_vocab, self.tgt_vocab)
        
    def len_filter(self, example):
        return len(example.src) <= MAX_LEN and len(example.tgt) <= MAX_LEN

    def len_filter_test(self, example):
        return len(example.src) <= MAX_LEN
    
    def print_dataset_info(self, data, src, tgt, input_vocab, output_vocab):
        print('20 tokens from input vocab:\n', list(input_vocab.stoi.keys())[:20])
        print('\n20 tokens from output vocab:\n', list(output_vocab.stoi.keys())[:20])
        print('\nnum training examples:', len(data.examples))
        item = random.choice(data.examples)
        print('\nexample train data:')
        print('src:\n', item.src)
        print('tgt:\n', item.tgt)
        print("\nSRC VOCAB LEN:",len(input_vocab))
        print("TRG VOCAB LEN:",len(output_vocab))
        print()


'''编码器'''
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        '''
            在__init__()函数初始化使用的网络
        '''
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, myinput, hidden):
        '''
            1.forward()函数接收输入的id序列, 送入Encoder, 编码返回最后
              一个token对应的隐藏状态用于初始化Decoder的隐藏状态
            2.如果要采用Attention, Encoder需返回所有token对应的隐藏状态
        '''
        embedded = self.embedding(myinput)
        embedded = embedded.view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self): 
        '''
            初始化隐藏层
        '''
        return torch.zeros(1, 1, self.hidden_size, device=config.device)


'''解码器'''
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        '''
            在__init__()函数初始化使用的网络
        '''
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        '''
            1.forward()函数以<sos>对应的id作为第一个输入, t时刻的输入为
              t-1时刻的输出, 解码出预测序列第t个token
            2.解码过程迭代至预测出<eos>这一token, 或者达到预设最大长度结束
            3.如果采用Attention, 需要在Decoder计算Attention Score
        '''
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.fc(output[0]))
        return output, hidden

    def initHidden(self):
        ''' 
            初始化隐藏层 
        '''
        return torch.zeros(1, 1, self.hidden_size, device=config.device)
    

'''Seq2Seq模型训练'''
def seq2seq(input_tensor, output_tensor, encoder, decoder,
            encoder_optimizer, decoder_optimizer, criterion,
            max_length=MAX_LEN, teacher_forcing_ratio=0.5):
    
    encoder_hidden = encoder.initHidden()     # 初始化编码器隐藏层
    encoder_optimizer.zero_grad()             # 将优化器的梯度设为0
    decoder_optimizer.zero_grad()
    input_length = input_tensor.size(0)       # 获取输入张量的长度，用于编码器循环
    output_length = output_tensor.size(0)     # 获取输出张量的长度，用于解码器循环
    encoder_outputs = torch.zeros(            # 创建空张量用于填充编码器输出
                        max_length, 
                        encoder.hidden_size, 
                        device=config.device)   
    loss = 0                                  # 损失变量
    
    for ei in range(input_length):            # 将输入张量传递进编码器
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor(             # 为解码器创建一个序列开始<sos>的张量
            [[TGT_VOCAB.stoi[SOS_TOKEN]]],
            device=config.device)
    decoder_hidden = encoder_hidden           # 将解码器隐藏状态设置为编码器的最终隐藏状态

    if random.random() < teacher_forcing_ratio:  # 决定我们是否使用 Teacher Forcing
        use_teacher_forcing = True  
    else:
        use_teacher_forcing = False

    for di in range(output_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()  # 从解码器输出历史中分离出来作为解码器输入
        loss += criterion(decoder_output, output_tensor[di].unsqueeze(0))
        if use_teacher_forcing:
            decoder_input = output_tensor[di]
        if decoder_input.item() == TGT_VOCAB.stoi[EOS_TOKEN]:   # 达到序列结尾<eos>
            break                                               

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.item() / output_length


'''执行迭代训练'''
def train(data_train, encoder, decoder, epoch_num, print_every=1000, 
               learning_rate=0.01, teacher_forcing_ratio=0.5):
    
    print(f'Running {epoch_num} epochs...')
    print_loss_total = 0
    print_loss_epoch = 0
    loss_ls = []                                                       # 记录损失的列表
    encoder_optim = optim.SGD(encoder.parameters(), lr=learning_rate)  # 定义编码器优化器
    decoder_optim = optim.SGD(decoder.parameters(), lr=learning_rate)  # 定义解码器优化器

    batch_iterator = torchtext.data.Iterator(                          # 批迭代器
        dataset=data_train, batch_size=config.batch_size,              # 批次大小设为1
        sort=False, sort_within_batch=True,
        sort_key=lambda x: len(x.src),
        device=config.device, repeat=False)
    criterion = nn.NLLLoss()                                           # 损失评价指标

    for e in range(epoch_num):
        batch_generator = batch_iterator.__iter__()               
        step = 0
        start = time.time()

        for batch in batch_generator: # 不使用批次，设批次为1是为了选择数据集中的第一条数据
            step += 1
            input_tensor, input_lengths = getattr(batch, 'src')    # 从批迭代器中获取输入
            output_tensor = getattr(batch, 'tgt')                  # 从批迭代器中获取输出
            input_tensor = input_tensor[0]
            output_tensor = output_tensor[0]

            loss = seq2seq(input_tensor, output_tensor, encoder, decoder,    # 训练模型
                           encoder_optim, decoder_optim, criterion, 
                           teacher_forcing_ratio=teacher_forcing_ratio)
            print_loss_total += loss
            print_loss_epoch += loss
            
            if step % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                loss_ls.append(print_loss_avg)
                print_loss_total = 0
                t = (time.time() - start) / 60
                print(f'step: {step}, avg loss: {print_loss_avg:.2f}, time: {t:.2f} min')
                start = time.time()
        
        print_loss_avg = print_loss_epoch / step
        print_loss_epoch = 0
        print(f'End of epoch {e+1}, avg loss {print_loss_avg:.2f}\n')

    torch.save(encoder.state_dict(), config.encoder_save_path)   # 存储训练好的编码器模型
    torch.save(decoder.state_dict(), config.decoder_save_path)   # 存储训练好的解码器模型
    return loss_ls


'''使用贪心搜索算法解码器来进行评估'''
def evaluate(encoder, decoder, sentence, max_length=MAX_LEN):
    with torch.no_grad():
        input_tensor = torch.tensor([SRC_VOCAB.stoi[word] for word in sentence], 
                                    device=config.device)
        input_length = input_tensor.size()[0]
        if input_length > max_length:
            input_length = max_length
        encoder_hidden = encoder.initHidden()
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=config.device)
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            try:
                encoder_outputs[ei] += encoder_output[0, 0]
            except:
                print(ei,input_length)
                break
        decoder_input = torch.tensor([[SRC_VOCAB.stoi[SOS_TOKEN]]], device=config.device)
        decoder_hidden = encoder_hidden
        decoded_words = []

        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            next_word = TGT_VOCAB.itos[topi.item()]
            decoded_words.append(next_word)
            if next_word == EOS_TOKEN:
                break
            decoder_input = topi.squeeze().detach()
        return decoded_words
    

'''随机评估数据集中的案例'''
def random_eval(data, encoder, decoder):
    print("\nRandom evaluating dev dataset:")
    for i in range(5):
        item = random.choice(data.examples)
        seq = item.src
        print(seq)
        words = evaluate(encoder, decoder, seq)
        print(' '.join(words))
        print()

'''评估测试集并将结果保存在txt文件'''
def eval_test(data, encoder, decoder, save_path):
    result = []
    for item in data.examples:
        string = ""
        seq = str(item.src)
        string += seq + "\n"
        words = evaluate(encoder, decoder, seq)
        pred = ' '.join(words)
        string += pred + "\n\n"
        result.append(string)
    with open(save_path,"w") as file:
        file.writelines(line for line in result)
    print('\nResult of test set successfully saved to \"'+save_path+'\"\n')


'''评估验证集的BLEU分数'''
class BLEU_Score():
    def __init__(self, data, encoder, decoder, ngram=4):
        print("\nGetting BLEU Score of test dataset:")
        self.data = data
        self.encoder = encoder
        self.decoder = decoder
        self.ngram = ngram
        src_unique = list(set([tuple(e.src) for e in self.data.examples[1:]]))
        self.BLEU_score = self.bleu(src_unique)

    def bleu(self, src_unique):
        '''
            返回BLEU分数
        '''
        BLEU_score = []
        for i, src in enumerate(src_unique):
            BLEU_score.append(self.get_score(src, self.ngram))
        print("BLEU Score:",np.mean(BLEU_score))
        return BLEU_Score

    def get_ref_list(self, src):
        '''
            寻找所有ref
        '''
        ref_list = []
        for e in self.data.examples:
            if e.src == list(src):
                ref_list.append(e.tgt)
        return(ref_list)

    def get_cand(self, src):
        '''
            返回预测的ref
        '''
        cand = evaluate(self.encoder, self.decoder, src)
        return(cand)

    def get_score(self, src, ngram):
        '''
            计算BLEU分数
        '''
        cand = self.get_cand(src)
        p_list = []
        ref_list = self.get_ref_list(src)
        
        for n in range(1,ngram + 1):      # 遍历n-gram
            ref_vocab_list = []
            for ref in ref_list:          # 为所有ref建立n-gram词典
                ref_vocab = dict()
                for i in range(len(ref)-n+1):
                    s = tuple(ref[i:(i+n)])
                    if ref_vocab.get(s):
                        ref_vocab[s] += 1
                    else:
                        ref_vocab.update({s:1})
                ref_vocab_list.append(ref_vocab)
            
            cand_vocab = dict()           # 为candidate建立n-gram词典
            for i in range(len(cand)-n+1):
                s = tuple(cand[i:(i+n)])
                if cand_vocab.get(s):
                    cand_vocab[s] += 1
                else:
                    cand_vocab.update({s:1})

            numerator = 0                 # 计算n-gram精度
            for s in cand_vocab:
                max_ref = max([ref_vocab.get(s) if ref_vocab.get(s) else 0 \
                               for ref_vocab in ref_vocab_list])
                max_ref = max_ref if max_ref else 0
                numerator += max_ref
            try:
                p = numerator/(len(cand)-n+1)
            except:
                p = 0
            if n > 1 and p == 0:
                p = p_list[-1]
            p_list.append(p)
        
        c = len(cand)                    # Brevity Penalty (BP惩罚)
        len_list = np.array([len(ref) for ref in ref_list])
        r = len_list[np.argmin(abs(len_list-c))]
        BP = max(1,np.exp(1-r/c))
        return BP * np.exp(np.mean(np.log(p_list)))


'''绘制训练集的损失图'''
def plot_loss(loss_ls, plot_save_path):
    plt.figure(1)
    plt.rcParams["figure.figsize"] = [12, 6]
    plt.rcParams["figure.autolayout"] = True
    plt.plot(loss_ls)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title("E2E NLG Loss")
    plt.savefig(plot_save_path)
    plt.show()


'''主函数'''
def main():
    # 分析数据
    analysis = Data_analysis()

    # 读取数据
    dataset = E2EDataset()
    train_set = dataset.train_data
    dev_set = dataset.dev_data
    test_set = dataset.test_data
    SRC_VOCAB = dataset.src_vocab
    TGT_VOCAB = dataset.tgt_vocab

    config = Config()
    is_train = True

    # 初始化编码器和解码器
    encoder = Encoder(input_size=config.input_dim, 
                      hidden_size=config.hidden_size).to(config.device)
    decoder = Decoder(hidden_size=config.hidden_size, 
                      output_size=config.output_dim).to(config.device)
    
    if os.path.exists(config.encoder_save_path) and os.path.exists(config.decoder_save_path):
        is_train = False
    
    if not is_train:
        print("Loading encoder and decoder model")
        encoder.load_state_dict(torch.load(config.encoder_save_path))
        decoder.load_state_dict(torch.load(config.decoder_save_path))

    # 训练集用于训练
    if is_train:
        loss_ls = train(data_train=train_set, 
                            encoder=encoder, 
                            decoder=decoder, 
                            epoch_num=config.max_epoch,
                            print_every=1000, 
                            learning_rate=config.lr, 
                            teacher_forcing_ratio=0.5)
    
    # 验证集用于评估和计算BLEU分数
    random_eval(dev_set, encoder, decoder)
    dev_bleu = BLEU_Score(data=test_set, 
                      encoder=encoder,
                      decoder=decoder,
                      ngram=4)
    
    # 将测试集的预测结果按要求保存在txt文件中
    eval_test(data=test_set, 
              encoder=encoder, 
              decoder=decoder, 
              save_path=config.save_data_path)
    
    # 绘制损失图像
    plot_loss(loss_ls=loss_ls,
              plot_save_path=config.plot_save_path)
    

if __name__ == '__main__':
    main()


    

