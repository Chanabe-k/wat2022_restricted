# Dementia_dialogue
- 大武先生の実験手伝い

## 環境構築メモ
- local version (`dementia`)
    - https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/
    - pyenvが怪しいらしいのでこれで環境構築した
    - `source dementia/bin/activate`
- server version (`dementia_cuda10`)
    - `qrsh -jc gpu-container_g1_dev -ac d=nvcr-pytorch-2003` # コンテナに入る
    - `source ./dementia_cuda10/bin/activate` # cuda10のvenv仮想環境に入る

## log
### 3/24
- 今ある対話アノテーションデータから「現在」「過去」「未来」を識別するclassifierを[Transformers](https://github.com/huggingface/transformers)の鈴木mさんの日本語BERTで作る
    - `pip install transformers==2.4.1`
        - ERROR
            -   running build_rust
            error: Can not find Rust compiler
            ----------------------------------------
            ERROR: Failed building wheel for tokenizers
            Failed to build tokenizers
            - Rustがない？みたいなこと言ってる
            - From sourceで install しても同じようなERRORを吐く
                - [Issue](https://github.com/huggingface/transformers/issues/2831)でver2.4.1を入れるといいよ！と書いてある

    - とりあえずQuick Startとかやる？
    - この前リツイートされていた[Twitterの使い方講座](https://twitter.com/huggingface/status/1205283603128758277)みる？
        - 完全にコピーしたものを`sample.py`として作成

    - torch > 1.0.0が必要なので[pytorch](https://pytorch.org/)をinstall
    - `pip install torch torchvision`

    - `python sample.py`
        - pathが間違ってると怒られた
            - `bert-base-japanese-whole-word-masking`だった
        - MeCabがないと怒られた
            - `pip install mecab-python3` 

### 3/29(Sun)
- 動いた

### 3/30(Mon)
- [(Part 2) tensorflow 2 でhugging faceのtransformers公式のBERT日本語学習済みモデルを文書分類モデルにfine-tuningする](https://tksmml.hatenablog.com/entry/2019/12/15/090900)
    - これを見つつ分類モデルをfine-tuningする

- 前処理として、
    1.  大武先生のデータを「発話文, 過去等のラベル」的なcsvに抽出
    2. Mecab(ipadic)で分割
    3. BertJapaneseTokeniserでid化

#### 1. 大武先生のデータを「発話文, 過去等のラベル」的なcsvに抽出
- `./data/annotationsのコピー2.csv`から`script`（実際の発話）と`time`（過去・現在・未来等のannotation）を抽出
    - ついでにidとfile_idも
    - csvって言ってるけどtsvだった
    - 文字コードがutf-16だった

- `annotations.csv`には全てのscriptに対してtimeが網羅的につけられているわけではない
    - 基準は？ -> 複数
        - 対象外フラグ（重複等でannotationの対象外とした）
        - time_impossible（時間の判定が困難）
            - このフラグたちがつけられていることを前提として学習するのか、それともそれらも加えて学習するのかどっち？
            - 対象外フラグはこの時間判定器にかける前につけられそうだが、「時間の判定が困難」は厳しいのでは？それも（鈴木さんタスクにおける「解答不可能」を判定する、みたいな感じで））時間判定器が出す一つの分類の出力として入れた方が良い？
            - そういう付け方ではない。むしろ「過去」とかが付いている者に対してflag=1を立てている

        - **intention = 発言、の場合、全てのannotationがつけられていない**
            - 過去等を判断する前に、まずintentionを判断する必要があるのでは？？
        - こういう時にどう実装すればいいか困ったらoption化しろって言ってた
        - のでargparseを導入し、どのannotationをつけるかoptionで選べるようにした
    
- `extract_data.py`というデータ抽出用のコードを作成
    - `python extract_data.py ./data/scripts_time.tsv -t `
    - `python extract_data.py ./data/scripts_time_intention.tsv -t -i`
    - をそれぞれ実行し、scriptとtimeだけついたデータ(scripts_time.tsv)、scriptとtime, intentionが付いたデータ(scripts_time_intention.tsv)を作成

- `tokenize_data.py`というデータtokenize用のコードを作成
    - scriptのMecab分割
    - timeラベルID化
        - timeラベルの異なり数は10（ID: 0~9）
            - `{'過去', '過去-最近', '最近（1か月以内）', '現在（状態、性質、考えなど）', '過去-現在（習慣など）', '未来（予定、予測、願望、仮定など）', '現在-未来', '最近-未来', '過去-未来', '最近-現在（習慣など）'}`
            - （そんなにあるのか......）
            - これ、分類できるのか？？
            - 「過去・現在・未来」の3つに集約したくない？ → これは大武先生らと相談する必要がありそう
                - （前まではそうだった気がするが.....、大武先生らにとっては「せっかくラベル増やしたのに.....」ということにならない？）
            - それぞれのラベル頻度を見る → 低頻度ラベルは無視しても良いかも？
            - intention = 発言に対するアノテーション？

- 方針的にはやはりこのままで良さそう
    - とりあえず10値分類で、`scripts_time.tsv`をtrain/dev等に分割してBERTを再学習させてみる
    - その結果を見せながら今年度初回ミーティングを企画する（4月下旬）

### 4/3(Fri)

#### tokenize_data.py
- それぞれのラベル頻度を見る
    - `label_freq:      Counter({'現在（状態、性質、考えなど）': 29626, '過去': 12065, '過去-現在（習慣など）': 5037, '最近（1か月以内）': 3749, '未来（予定、予測、願望、仮定など）': 2073, '最近-現在（習慣など）': 182, '過去-最近': 117, '現在-未来': 74, '過去-未来': 58, '最近-未来': 13})`
    - グラフで可視化したいのでmatplotlibをinstall
    - `pip install matplotlib`

    - 時間ラベルの頻度分布を調べるのを関数化した
        - `count_label_freq()`
            - [plt.savefig(bbox_inches = tight)](http://virsalus.hatenablog.com/entry/2015/01/19/120931)
            - [mpl.rcParams['font.family'] = 'Meiryo']()
            - [plt.xticks(rotation = 270)](https://www.delftstack.com/ja/howto/matplotlib/how-to-rotate-x-axis-tick-label-text-in-matplotlib/)

    - 松田さん「とりあえずは，多い方から5ラベルだけを相手にする方向でいいかもしれませんね．ほかはどこかにマージするか，無視するか，そのあたりは大武チームと相談しましょう．」
        - わかる
        - とりあえず10値分類と5分類（削り）を試してみるか

### 4/6(Mon)
#### tokenize_data.pyの続き
- Mecab分割をしたい
    - Mecab-python, うまく動かすのむずいんだよな〜と思ってたけど、BertJapaneseTokenizerでいける？？
- 上のリンク先のtensorflowを用いたdata tokenizeコードをcopy

### 4/7(Tue)
#### 昨日に引き続きtokenize_data.py
- めんどくさいから先にMeCab分割がちゃんどできるかみたい
- そもそも1行ごとの処理ではなく、pandasで一気に処理した方が良さそう（tfを用いたデータ分割がDataFrame上でで行われているため）
    - `pip install pandas`

- **train / dev / testの分割をfile_id単位で行う**
    - から、上のリンク先のコードをそんな単純に使えるわけではない
        - file_id フィールドがそれっぽいですね．226セッションあるのですね．`200 / 13 / 13` くらいか，もうすこし dev / test が多い `150 / 38 / 38` か，それくらいでしょうか．by 松田さん

#### 松田さんと打ち合わせ
- 最初から5ラベルでやってみる

- 他のannotationラベルについての分布可視化もしておく？（なんかやってる感を出す.....）
    - 理想はRAIDENで学習回してるときにこの分布可視化 & スライド作成

### 4/8(Wed)
- とりあえず方言ジャーナルを無理やりfixさせてこっちに取り掛かる

#### extract_data.pyでid化したtimelabel列を作成する
- `./data/script_time.tsv`をそういうデータに置き換える
    - ついでに、ここで5値に減少させる......？
    - `{'過去': '0', '過去-最近': '1', '最近（1か月以内）': '2', '現在（状態、性質、考えなど）': '3', '過去-現在（習慣など）': '4', '未来（予定、予測、願望、仮定など）': '5', '現在-未来': '6', '最近-未来': '7', '過去-未来': '8', '最近-現在（習慣など）': '9'}`
    - 削るのは`'最近-現在（習慣など）': 182, '過去-最近': 117, '現在-未来': 74, '過去-未来': 58, '最近-未来': 13`
    - `1, 6, 7, 8, 9`

`!head ./data/script_time.tsv`
```id	file_id	script	time    time_id
148177	1	私以外はみんな東京の方なんですかねみなさん	現在（状態、性質、考えなど）	3
3	1	中央区ですはい	現在（状態、性質、考えなど）	3
4	1	私文京区です	現在（状態、性質、考えなど）	3
142715	1	私は千葉県です。	現在（状態、性質、考えなど）	3
```
- ちょうどきりの良い数字になった

#### tokenize_data.py
- updateされた`./data/script_time.tsv`のtime_id列をlabelとして参照
- とりあえずload_dataset()までは動いたっぽい？

### 4/21(Tue)
- マジでしばらく放置してたな.......

- RAIDENに実行環境を整える必要がある
    - https://io-lab.esa.io/posts/1047
    - 色々面倒っぽい.......

- 環境はcondaで作るのが良い？
    - 清野さんはpyenvで作っている？でも結構前の記事だからな.....
- コンテナでやるのが良い？？
    - 人に聞くのが一番速そうなきもしないでもない......

- https://files.esa.io/uploads/production/attachments/4896/2018/04/23/18306/c5294517-86c3-4ad6-8a54-b39319ed9414.pdf
    - 最初はこれを見てやるのが一番いいらしい
    - Hands on #2あたりが参考になるか？
    - なんか変だと思ったら、`after login to the container`って書いてある
        - zshに変えたけど、普通にbashでやった方が楽そう...
            - だけど、.bashrcをrmしてしまった.......
            - [./bash_projileや.bashrcの大元は/etc/skelにある](https://qiita.com/shyamahira/items/260862743e4c9794b5d2)のでそこからcopyしてきて解決
    
    - esaの「fairseqのインストールまで」って所を見る
        - fairseq->transformerにすれば良いわけだし行けそう？
        - `# コンテナに入り，環境変数などの設定をする`の部分は実行済
        - `#cuda10仮想環境の有効化をする`あたりから変える必要がありそう
            - とりあえず今あるdementia環境に`source dementia/bin/activate`でログインした上で、`pip install torch torchvision`をしてみる（Linux, cuda10を選んだらそれになったがlocalでやった時と同じでは.....？）
                - なんかめっちゃRetryしてる......
                - 最終的に`Could not find a version that satisfies the requirement torch (from versions: )No matching distribution found for torch`と怒られた
            - やっぱりserver上で新しく環境作った方が良いんじゃないか......？
                - 一から「dementia_cuda10」環境を作ろうとしたら、`python3-venv`がないと怒られた
                - んん......
                - condaで作るか...？
                - condaのinstallから始まる
                    - 清野さんはminicondaらしいのでminicondaをinstallする
                    - これ最新のだとpy=3.7になるっぽいけどそれでいいのかな.......
                - miniconda installしたぞと思ってconda createとかやってもnot foundと怒られる
                - InstallationにPATHに通せと書いてあるからPATHを見たけど、思った以上にPATH色々通ってるな？？？？？→よく読んだら「install shell scriptは自動的にpathを通してくれます」って書いてないか？
                    - condaとかちゃんと書いてない？？
                    - あ、`.bashrc`にminicondaへのpathを書いてくれてるのね......

                - `conda create -n cuda10.0 py=3.6`を試しにやってみる
                    - py=3.6でいいのか......
                    - `Collecting package metadata`から動かない.....
                    - え？？anacondaのHTTPに繋がないといけないの？？
                        - 「コンテナの中でインターネットに繋ぐ」をやんないといけないってこと？？
                        - なんなん？？
                        - 見た感じ既に「base」という環境に入ってるぽいのでもうそこにinstallしていっちゃう？
                        - conda install pytorch torchvision cudatoolkit=10.0 -c pytorch

### 4/23(Tue)
- 鈴木さんと雑談
    - 鈴木さんはprojectごとのディレクトリにpython3のvenvで仮想環境作ってる
        - ログインノードだからpip等が動かなかった
        - GPUノードだと4時間しかインタラクティブにできないので、長時間ならジョブを投げないといけない
    - esaに書いてある`setup.py`をsourceにしとく必要がある？（これはそのままコピーして大丈夫）
    - IP adressがCPUノードとGPUノードで違う
        - For PPCって書いてるのがCPUノード
    - 「コンテナの中でインターネットに繋ぐ」 & 「コンテナ内でvenv環境を使う」試してみる！ 

### 4/24(Fri)
- 「コンテナの中でインターネットに繋ぐ」をやる
    - `setup.py`書いてbash_profileに実行する旨書く
        - `Collecting package metadata...`はいい感じに行ってるっぽいが、py=3.6とか3.7と指定してもPackagesNotFoundErrorと言われる

- 「コンテナ内でvenv環境を使う」
    - `.bashrc`に書いてあるcondaへのPATH通しを一旦コメントアウトして、venv環境を使う
    - `pip install torch torchvision --user`を試してみる
        - `Permission denied error`が出た
        - `--user` optionが必要
        - なんかこれまでつまづいていたところはある程度いけてるっぽいが、よく読むとsite-packagesとかが「python3.5」と書いてある
        - なんで？

    - globalに参照するpython3がpython3.5.2っぽい？？
    - え〜〜
        - やっぱminicondaの環境使う？
        - minicondaの(base)環境に入っても、python3で実行されるのはpython3.5.2なのだが......
        - `~/miniconda3/bin/python3.7`を実行するとPython3.7.6が起動される（それはそう）

        - [PYTHONUSERBASE](https://qiita.com/ronin_gw/items/cdf8112b61649ca455f5)なる環境変数があるらしい
            - 設定したが、なぜかminiconda3の中にもpython3.5があり、頑なにpython3.7にinstallしてくれない
            - なんで？？？？

    - `conda list`を実行すると、pythonは3.7.6だと書かれている
        - 普通にpy=とか指定しなければいいのでは？というか、たぶんpython=3.6とかならいけた？？
        - いけた〜〜〜〜

#### サーバ仮想環境構築(dementia_cuda10)
- condaによる格闘の跡
    - `conda activate cuda10.0`
    - `pip install tensorflow` （現versionはgpuとか区別しなくて良いっぽい......？）
        - cudaにPATH通す奴は...？
        - ていうかあれ、またpython3.5の環境にinstallしてない.....？
        - condaでinstallしたら良い.....？
    - `conda install pytorch torchvision cudatoolkit=10.0 -c pytorch`
    - `conda install -c anaconda tensorflow-gpu`
        - tensorflow==2.0.0 installできてるし、大丈夫そうではある.....
    - `conda install torchvision`
        - build py36_0とも書いてるし大丈夫そう......？
    - `conda install transformers`
        - これはPackagesNotFoundErrorが出る
    - `home/src`directoryを新しく作り、そこにtransformers　repositoryを置く
        - いや、結局`pip install .`を実行できないと厳しい......？

- `venv`格闘に戻る
    - python3.5になっちゃうやつを鈴木mさんに聞く
        - コンテナに入っているpythonが3.5になってしまっている
    - `nvcr-pytorch-2003`のコンテナ（python3.6)内でやると良い？
        - `cat containor-info`
    - やった〜〜〜できた〜〜〜

- `dementia_cuda10`環境設定
    - `pip install transformers==2.4.1`
    - `pip install https://download.pytorch.org/whl/cu100/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl`
    - `pip install torchvision`
    - `pip install mecab-python3`

- **`python sample.py`が動いた！！！！！！！！**
    - もう終わっていいか（ダメ）

### 5/19(Tue) 
- 1ヶ月くらい経とうとしていた......これはいかん

- 科研費奨励費のアレ
    - 「クラウドソーシングで人手評価するぞ！」と言っていたけど、個人情報とか大丈夫なんだっけ？
    - [理研の研究倫理のやつ](https://www.riken.jp/medialibrary/riken/about/reports/ethics/ethics-bylaw_20190723.pdf)
        - 「ヒトゲノム・遺伝子解析研究以外の研究に係る個人情報等の保護」の第7条とかがそれっぽい
            - 匿名化とかすれば基本的には良い
            - けど住んでるところがだいたいわからなくもないので、その辺も少し必要かも

- やることを整理しよう
    - 土日で学習を回したい
        - 前処理の完遂
            - MeCabをかける
            - japanese-BERTの中に入っているtokenizerで分ける
        - 評価をどうするか（データのsplit問題）
            - **train / dev / testの分割をfile_id単位で行う**
            - file_idで分割
            - 226セッションある → `150 / 38 / 38` で分割する

- あのサイトに従ってやらない方が良い気がしてきた, 自分でゆっくりやって行った方が良さそう -> [これ](https://qiita.com/nekoumei/items/7b911c61324f16c43e7e)は参考になるかも
    - python ./tokenize_data.pyを実行しようとしたらエラーをtensorflowがないと言われたので、`pip install tensorflow-gpu`をしてみる
        - なんかPATHが通ってないとかって怒られたから一応`/uge_mnt/home/abe-k/.local/bin`をexportでPATHに通す
    - 「お前は何を言っているんだ」的なエラーが出た
        - > OSError: Model name 'bert-base-japanese' was not found in tokenizers model name list (bert-base-japanese, bert-base-japanese-whole-word-masking, ...)
        - 松田さんから「modelの置き場所が変わったらしい」との情報、upgradeしたら動くようになった

- `tokenize_data.py`tokenizeできるようになった
    - tokenizeしたデータを持っていた上で、train, dev, testに分けたい

#### 明日やること
- 早くtrain, dev, testに分けようね

- modelのfine-tuningをどうするかに関してはなんか色々なサイトが出てきて混乱している。皆自己流でやってんな......
    - GLEU taskを解く流れにぬるっと新しいtaskを挿入する、ということをやっている人もいる(https://qiita.com/kenta1984/items/7f3a5d859a15b20657f3)、がGLEUの概念を崩壊させていそうなのであんまりやりたくない（簡単そうだけど.....）

- （余談）rikenサーバの端末のプロンプトを変えたい（未解決）
    - https://qiita.com/wildeagle/items/5da17e007e2c284dc5dd
        - `~/.bashrc`に書いたら、仮想環境をactivateするまではいいのだがactivateした後ダメになる
        - う〜む

- （余談2）kiyonoさんのスライドに従ってSFTPしたい（**解決**）
    - 曰く、使っているソフト名 + SFTPと検索すれば良い→でてきたのが[コレ](https://qiita.com/ishimasar/items/1324af16e19a59b220d3) 
    - simpleな`sample_sftp`というディレクトリに関してはうまく行ったが、`dementia_dialogue`に関してはうまくいかない...
    - と思ったら, portの問題だったっぽい？
        - めっちゃ頑張って`dementia_cuda10`の設定とかdownloadしてくれてるっぽい...これ大丈夫か......

### 5/20(Wed)

#### データ分割
- 効率的にやるのは諦めて、とりあえず自分でわかるようにやっていこう...

- tokenizedしたscript列を追加した`./scripts_time.tsv.tok`をtmpディレクトリに一旦保存→それを分割、というふうにする

- `./scripts_time.tsv.tok`を読み込んで、file_idで分割
    - 226セッションある → `150 / 37 / 37` で分割する
        - train: 1 ~ 150
        - dev: 151 ~ 189
        - test: 190 ~ 226

- セッションごとに分割 → 実際の文数は以下のようになった
```
11019 ./data/test.tok
32816 ./data/train.tok
8715 ./data/valid.tok
----------------------
52550 total
```


## TODAY
### 5/22(Fri)
#### モデル作成
- https://qiita.com/kenta1984/items/7f3a5d859a15b20657f3
- ちょっと邪道だが、これを参考にしてみる
    - transformersのglue.pyとmetrics/__init.pyをいじる
    - 以下を実行
    ```run.sh
    DATA_DIR=/home/abe-k/dementia_dialogue/dementia_dialogue/data/
    OUTPUT_DIR=/home/abe-k/dementia_dialogue/dementia_dialogue/output

    python ~/src/transformers/examples/text-classification/run_glue.py \
    --data_dir=$DATA_DIR \
    --model_type=bert \
    --model_name_or_path=bert-base-japanese-whole-word-masking \
    --task_name=original \
    --do_train \
    --do_eval \
    --output_dir=$OUTPUT_DIR
    ```

- `run_glue.py`はgit cloneしたリポジトリの中のexampleの部分にあるっぽい
    - `home/abe-k/src/`以下にinstall する（version管理のため）
    - `pip install -r ./examples/requirements.txt`

    - 上のやつを動かす前に、そもそも普通のglueがうまくいくのか試したい...
        - `transformers/data`にglue（wnli）のデータをinstallしてみる
         - localに一旦落としてから`rsync`でサーバにあげようかと思ったけど、とりあえず`wget https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FMNLI.zip?alt=media&token=50329ea1-e339-40e2-809c-10c40afff3ce`を試してみる → フリーズしている & ダウンロード具合が表示されない。**wgetは使わない方が良さそう**
        - **rsyncで送る時、サーバ側のpwdで出てくる`uge_mnt`はいらない**

    - 以下を実行するも、FileNotFoundErrorが出る
    ```python examples/text-classification/run_glue.py \
    --data_dir=/data/WNLI \
    --model_type=bert \
    --model_name_or_path=bert-base-uncased \
    --task_name=wnli \
    --do_train \
    --do_eval \
    --output_dir=output/
    ```
    > FileNotFoundError: [Errno 2] No such file or directory: '/data/WNLI/cached_train_BertTokenizer_128_wnli.lock'
    - 検索しても、1件もHITしない。あれえ.....？
        - examples/text-classificationの中のREADME.mdを見たら、xnliにはxnli用のscriptがあったり、他のglueタスクにもそれぞれsnippetがあったりしたのでそれを実行した方が良さそう。
    
    - ↑を実行したら結局src/transformersの中身を実行しているので、それをいじる
        - glue_tasks_num_labelsとかと言われている

> 05/22/2020 05:46:39 - WARNING - __main__ -   Process rank: -1, device: cpu, n_gpu: 0, distributed training: False, 16-bits training: False 
    - GPU使ってないな.....

Traceback (most recent call last):
  File "/uge_mnt/home/abe-k/src/transformers/examples/text-classification/run_glue.py", line 108, in main
    num_labels = glue_tasks_num_labels[data_args.task_name]
KeyError: 'original'

- 以下は上のエラーからassertされたものだから一旦置いておく
During handling of the above exception, another exception occurred:
Traceback (most recent call last):
  File "/uge_mnt/home/abe-k/src/transformers/examples/text-classification/run_glue.py", line 228, in <module>
    main()
  File "/uge_mnt/home/abe-k/src/transformers/examples/text-classification/run_glue.py", line 111, in main
    raise ValueError("Task not found: %s" % (data_args.task_name))
ValueError: Task not found: original

- 直したが、ipdbで確かめてみてもglue_tasks_num_labelsにoriginalが入っていない🤔
> ipdb> glue_tasks_num_labels                                                                                                                                 
{'cola': 2, 'mnli': 3, 'mrpc': 2, 'sst-2': 2, 'sts-b': 1, 'qqp': 2, 'qnli': 2, 'rte': 2, 'wnli': 2}

- 考えられうる場所（仮想環境`dementia_cuda10`のlib or git cloneしたリポジトリ）は`original`を付け加えたと思うんだけど、`import transformers`はどこをみてるんだ？
    - 「text-classification内」は見てるけど、src/transformersは見てなさそう？
ipdb> sys.path                                                                                                              
['/home/abe-k/dementia_cuda10/lib/python3.6/site-packages', '/uge_mnt/home/abe-k/src/transformers/examples/text-classification', '/opt/conda/lib/python36.zip', '/opt/conda/lib/python3.6', '/opt/conda/lib/python3.6/lib-dynload', '', '/uge_mnt/home/abe-k/.local/lib/python3.6/site-packages', '/opt/conda/lib/python3.6/site-packages', '/opt/conda/lib/python3.6/site-packages/IPython/extensions', '/uge_mnt/home/abe-k/.ipython']
`/uge_mnt/home/abe-k/.local/lib/python3.6/site-packages`ここにもtransformersある、これをimportしてるかも
    - これをimportしてた

- 動いたけど、またError
> Traceback (most recent call last):
  File "/uge_mnt/home/abe-k/src/transformers/examples/text-classification/run_glue.py", line 230, in <module>
    main()
  File "/uge_mnt/home/abe-k/src/transformers/examples/text-classification/run_glue.py", line 139, in main
    train_dataset = GlueDataset(data_args, tokenizer=tokenizer) if training_args.do_train else None
  File "/uge_mnt/home/abe-k/.local/lib/python3.6/site-packages/transformers/data/datasets/glue.py", line 111, in __init__
    output_mode=self.output_mode,
  File "/uge_mnt/home/abe-k/.local/lib/python3.6/site-packages/transformers/data/processors/glue.py", line 64, in glue_convert_examples_to_features
    examples, tokenizer, max_length=max_length, task=task, label_list=label_list, output_mode=output_mode
  File "/uge_mnt/home/abe-k/.local/lib/python3.6/site-packages/transformers/data/processors/glue.py", line 136, in _glue_convert_examples_to_features
    labels = [label_from_example(example) for example in examples]
  File "/uge_mnt/home/abe-k/.local/lib/python3.6/site-packages/transformers/data/processors/glue.py", line 136, in <listcomp>
    labels = [label_from_example(example) for example in examples]
  File "/uge_mnt/home/abe-k/.local/lib/python3.6/site-packages/transformers/data/processors/glue.py", line 131, in label_from_example
    return label_map[example.label]
KeyError: '3'

- これって、glueで想定される多値ラベルよりも多いから...？
    - 頑張ればどうにかできそう？
    - Original_Processorの, get_label部分を["0", "1"] → ["0", ~, "5"]に拡張
    
