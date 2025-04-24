# import akshare as ak

# stock_comment_detail_scrd_focus_em_df = ak.stock_comment_detail_scrd_focus_em(symbol="601020")
# print(stock_comment_detail_scrd_focus_em_df)

# import akshare as ak

# stock_comment_detail_zhpj_lspf_em_df = ak.stock_comment_detail_zhpj_lspf_em(symbol="601020")
# print(stock_comment_detail_zhpj_lspf_em_df)

# import akshare as ak

# stock_comment_detail_zlkp_jgcyd_em_df = ak.stock_comment_detail_zlkp_jgcyd_em(symbol="601020")
# print(stock_comment_detail_zlkp_jgcyd_em_df)

# import akshare as ak

# stock_comment_detail_scrd_desire_em_df = ak.stock_comment_detail_scrd_desire_em(symbol="601020")
# print(stock_comment_detail_scrd_desire_em_df)

# import akshare as ak

# stock_comment_detail_scrd_desire_daily_em_df = ak.stock_comment_detail_scrd_desire_daily_em(symbol="601020")
# print(stock_comment_detail_scrd_desire_daily_em_df)

import akshare as ak

stock_news_main_cx_df = ak.stock_news_main_cx()
print(stock_news_main_cx_df)

import jieba
from collections import Counter
import pandas as pd

# 将新闻标题合并为一个字符串
news_text = ' '.join(stock_news_main_cx_df['tag'].tolist())

# 使用jieba进行分词
words = jieba.cut(news_text)

# 过滤掉停用词和单个字符的词
stop_words = {'的','了','在','是','和','有','都','对','从','到','与','及','把','被','让','向','给','但','为','以','能','将','就','等','要','这','那','也','着','并','很','再','或','某','于','如','所','才','吧','只','而','已','却','还','比','情','去','她','说','据','他','看','据悉','表示','称'}
filtered_words = [word for word in words if word not in stop_words and len(word) > 1]

# 统计词频
word_counts = Counter(filtered_words)

# 获取前10个高频词
top_10_words = word_counts.most_common(100)

# 转换为DataFrame并打印
df_words = pd.DataFrame(top_10_words, columns=['词语', '出现次数'])
print("\n新闻标题中出现频率最高的10个词语:")
print(df_words)
# 将结果保存到CSV文件
df_words.to_csv("news_word_frequency.csv", index=False, encoding='utf-8-sig')
print("\n词频统计结果已保存到 news_word_frequency.csv")
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# 将词频数据转换为字典格式
word_freq_dict = dict(zip(df_words['词语'], df_words['出现次数']))

# 创建词云对象
wc = WordCloud(
    font_path='SimHei',  # 使用黑体字体以正确显示中文
    width=1200,          # 设置宽度
    height=800,          # 设置高度
    background_color='white', # 设置背景颜色
    max_words=100,       # 最多显示词数
    max_font_size=150    # 字体最大值
)

# 生成词云
wc.generate_from_frequencies(word_freq_dict)

# 创建图表
plt.figure(figsize=(15,10))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')  # 不显示坐标轴
plt.title('新闻热词云图', fontsize=20, pad=20, fontproperties='SimHei')

# 保存图片
plt.savefig('news_wordcloud.png', dpi=300, bbox_inches='tight')
print("\n词云图已保存为 news_wordcloud.png")

# 显示图片
plt.show()
