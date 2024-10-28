from __future__ import print_function
import json
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import SpTokenizer
from bert4keras.snippets import sequence_padding, open
from bert4keras.snippets import AutoRegressiveDecoder
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# 基本参数
max_c_len = 256
max_t_len = 32

# 模型路径
config_path = '/root/autodl-tmp/t5_in_bert4keras/mt5/mt5_base_config.json'
checkpoint_path = '/root/autodl-tmp/t5_in_bert4keras/mt5/mt5_base/model.ckpt-1000000'
spm_path = '/root/autodl-tmp/t5_in_bert4keras/mt5/sentencepiece_cn.model'
keep_tokens_path = '/root/autodl-tmp/t5_in_bert4keras/mt5/sentencepiece_cn_keep_tokens.json'

# 加载分词器
tokenizer = SpTokenizer(spm_path, token_start=None, token_end='</s>')
keep_tokens = json.load(open(keep_tokens_path))

t5 = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    keep_tokens=keep_tokens,
    model='t5.1.1',
    return_keras_model=False,
    name='T5',
)

encoder = t5.encoder
decoder = t5.decoder
model = t5.model

class AutoTitle(AutoRegressiveDecoder):
    """seq2seq解码器"""
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        c_encoded = inputs[0]
        return decoder.predict([c_encoded, output_ids])[:, -1]

    def generate(self, text, topk=1):
        c_token_ids, _ = tokenizer.encode(text, maxlen=max_c_len)
        c_encoded = encoder.predict(np.array([c_token_ids]))[0]
        output_ids = self.beam_search([c_encoded],topk)
        return tokenizer.decode([int(i) for i in output_ids])

autotitle = AutoTitle(start_id=0, end_id=tokenizer._token_end_id, maxlen=32)

# 加载最佳模型权重
model.load_weights('./best_model.weights')

def predict_title(content):
    """预测给定内容的标题"""
    return autotitle.generate(content)



# 示例用法
content_to_predict = "	提出了一种新的保细节的变形算法,可以使网格模型进行尽量刚性的变形,以减少变形中几何细节的扭曲.首先根据网格曲面局部细节的丰富程度,对原始网格进行聚类生成其简化网格;然后对简化网格进行变形,根据其相邻面片变形的相似性,对简化网格作进一步的合并,生成新的变形结果,将该变形传递给原始网格作为初始变形结果.由于对属于同一个类的网格顶点进行相同的刚性变形,可在变形中较好地保持该区域的表面细节,但分属不同类的顶点之间会出现变形的不连续.为此,通过迭代优化一个二次能量函数,对每个网格顶点的变形进行调整来得到最终变形结果.实验结果显示,该算法简单高效,结果令人满意."
predicted_title = predict_title(content_to_predict)
print(f"input：{content_to_predict}")
print(f"output：{predicted_title}")