# Deep Learning - Speech Recognition報告
## Deep Learning@NTUT, 2020 Fall報告
Taiwanese Speech Recognition using End-to-End Approach


- 學生: 郭靜
- 學號: 108598068

---

## 做法說明
1. 將kaggle下載的.wav檔(train/test)，利用轉檔套件sox，轉成16 kHz sampling，signed-integer，16 bits的格式，並且將檔名排序
2. 將train-toneless.csv中的每一行依序輸出成.txt檔，並且檔名依照音檔命名
3. 再將轉檔過後的.wav，轉成python可以讀取的.pkl檔
4. 定義模型，使用一層的GRU網路模型
5. 訓練模型，1500次以前，每100個epoch儲存一次模型，1500次以後，每20次存一個模型
6. 測試模型
7. 輸出與sample.csv同樣格式的csv檔
---

## 程式方塊圖與寫法

![](https://i.imgur.com/qK8X7X6.png)



#### 將kaggle下載的.wav檔(train/test)，利用轉檔套件sox，轉成16 kHz sampling，signed-integer，16 bits的格式，並且將檔名排序
```
import os
from os import listdir

files = listdir('./test_org')

for f in files:
    old_f = f
    f = '%04d' % int(f.replace('.wav', ''))
    os.system('sox ./test_org/' + old_f + ' -r 16000 -e signed-integer -b 16 test/wav48/p225/p225_' + f + '.wav')
```


#### 將train-toneless.csv中的每一行依序輸出成.txt檔，並且檔名依照音檔命名
```
import csv
import numpy as np

with open('sample.csv', newline='', errors='ignore') as csvfile:
	rows = csv.reader(csvfile)
	for row in rows:
		a=row[0]
		b=row[1]
		a = '%03d' % int(a)
		txt_path='./test/txt/p225/' + 'p225_' + a + '.txt'
		print(txt_path)
		f=open(txt_path, 'w')
		f.write(b)
		f.close()
```

#### 再將轉檔過後的.wav，轉成python可以讀取的.pkl檔
```
for filename in files:
    try:
        target_text = open(filename.replace('wav48', 'txt').replace('.wav', '.txt'), 'r').read().strip()
        speaker_id = extract_speaker_id(filename)
        audio = read_audio_from_filename(filename, self.sample_rate)
        obj = {'audio': audio,
               'target': target_text,
               FILENAME: filename}
        cache_filename = filename.split('/')[-1].split('.')[0] + '_cache'
        tmp_filename = os.path.join(cache_dir, cache_filename) + '.pkl'

    with open(tmp_filename, 'wb') as f:
        dill.dump(obj, f)
        print('[DUMP AUDIO] {}'.format(tmp_filename))
    if speaker_id not in self.metadata:
        self.metadata[speaker_id] = {}
    sentence_id = extract_sentence_id(filename)
    if sentence_id not in self.metadata[speaker_id]:
        self.metadata[speaker_id][sentence_id] = []
    self.metadata[speaker_id][sentence_id] = {SPEAKER_ID: speaker_id,
                                              SENTENCE_ID: sentence_id,
                                              FILENAME: filename}
    except librosa.util.exceptions.ParameterError as e:
        print(e)
        print('[DUMP AUDIO ERROR SKIPPING FILENAME] {}'.format(filename))
```


#### 定義模型，我使用一層的GRU網路模型
```
def get_a_cell():
    return tf.nn.rnn_cell.GRUCell(num_hidden)

stack = tf.contrib.rnn.MultiRNNCell([get_a_cell() for _ in range(1)], state_is_tuple=True)
outputs, _ = tf.nn.dynamic_rnn(stack, inputs, seq_len, dtype=tf.float32)
```

#### 訓練模型，1500次以前，每100個epoch儲存一次模型，1500次以後，每20次存一個模型
```
if ((curr_epoch+1)%100==0) or (curr_epoch+1 > 1500 and (curr_epoch+1)%20==0):
    save_path = saver.save(session, modelpath+'/'+str(curr_epoch), global_step=curr_epoch+1)
    print("save model to ", save_path)
```

#### 測試模型
```
with tf.Session(graph=graph) as session:
    # load model        
    saver = tf.train.Saver(max_to_keep=None)

    if os.path.exists('./model/checkpoint'):
        saver.restore(session, './model/1919-1920')
    else:
        init = tf.global_variables_initializer()
        session.run(init)

    for curr_epoch in range(num_epochs):
        for batch in range(num_batches_per_epoch):
            val_inputs, val_targets, val_seq_len, val_original = next_batch(train=False)
            val_feed = {inputs: val_inputs,
                    targets: val_targets,
                    seq_len: val_seq_len}

            d = session.run(decoded[0], feed_dict=val_feed)
            decode_batch(d, val_original, phase='validation')
```

#### 輸出與sample.csv同樣格式的csv檔
```
with open('pre.txt', 'r') as in_file:
    stripped = (line.strip() for line in in_file)
    lines = (line.split(',') for line in stripped if line)
    with open('pre.csv', 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(['text'])
        writer.writerows(lines)

df = pd.read_csv('pre.csv')
df.index = np.arange(1, len(df) + 1)
df.index.names = ['id']
df.to_csv('CTC_result.csv')
```

---

## 畫圖結果分析

epochs = 2000
num_hidden = 1024
batch_size = 16


* LER Loss
val_ler的震盪蠻大的，測試模型我選擇loss平均較小的來測試
![](https://i.imgur.com/opuzh9q.png)
* Cost
![](https://i.imgur.com/FgV3Pj6.png)
* Cost的局部圖，選擇測試模型我也會選擇平均cost較小的階段之模型
![](https://i.imgur.com/1Gy7U94.png)


---

## 討論預測值誤差很大的，是怎麼回事？
1. 一開始我使用一層的LSTM網路模型，結果沒有很好。
2. 可能網路中的隱藏神經元太少。

---

## 如何改進？
1. 將LSTM改成GRU的網路模型之後，結果比較好。
2. 將網路中的隱藏神經元調整為1024。

---
