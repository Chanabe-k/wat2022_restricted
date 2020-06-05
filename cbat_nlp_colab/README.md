# Dementia_dialogue
- 大武先生の実験手伝い

## 環境構築メモ
- https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/
    - pyenvが怪しいらしいのでこれで環境構築した
- サーバ仮想環境 (`cuda10`)
    - python3.6, transformers==2.10, **cuda10.2**
    - RAIDENで使うとき
        - `qrsh -jc gpu-container_g1_dev -ac d=nvcr-pytorch-2003` # python3.6が使えるpytorchコンテナに入る
        - `source /fefs/opt/dgx/env_set/common_env_set.sh`
        - `source /fefs/opt/dgx/env_set/nvcr-tensorrt-1901-py3.sh` # なんかpytorch-2003-py3.shにしたらうまく行かない。これでいいならこれでいいのでは
        - `/usr/local/bin/nvidia_entrypoint.sh`
        - `source ./cuda10/bin/activate` # cuda10のvenv仮想環境に入る

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

- 次のエラー（エラー100本knockみたいになってきた）
    > Traceback (most recent call last):
    File "/uge_mnt/home/abe-k/src/transformers/examples/text-classification/run_glue.py", line 230, in <module>
        main()
    File "/uge_mnt/home/abe-k/src/transformers/examples/text-classification/run_glue.py", line 140, in main
        eval_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="dev") if training_args.do_eval else None
    TypeError: __init__() got an unexpected keyword argument 'mode'
    これはgit cloneしたtransformersの`run_glue.py`の話なので、./local/libの中のtransformersとこの`run_glue.py`の不整合かもしれない
    - ./local内のGlueDatasetはevaluate=False, という形になっている→ これをTrueにするようにすればOK?
        - それぞれ`mode="dev"`, `mode="test`となってるんだけどどっちも`evaluate=True`でいいんだろうか...

- 次のエラー
> raceback (most recent call last):                                                                                                  | 0/4102 [00:00<?, ?it/s]
  File "/uge_mnt/home/abe-k/src/transformers/examples/text-classification/run_glue.py", line 230, in <module>
    main()
  File "/uge_mnt/home/abe-k/src/transformers/examples/text-classification/run_glue.py", line 162, in main
    model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
  File "/uge_mnt/home/abe-k/.local/lib/python3.6/site-packages/transformers/trainer.py", line 415, in train
    tr_loss += self._training_step(model, inputs, optimizer)
  File "/uge_mnt/home/abe-k/.local/lib/python3.6/site-packages/transformers/trainer.py", line 506, in _training_step
    outputs = model(**inputs)
  File "/opt/conda/lib/python3.6/site-packages/torch/nn/modules/module.py", line 550, in __call__
    result = self.forward(*input, **kwargs)
  File "/uge_mnt/home/abe-k/.local/lib/python3.6/site-packages/transformers/modeling_bert.py", line 1161, in forward
    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
  File "/opt/conda/lib/python3.6/site-packages/torch/nn/modules/module.py", line 550, in __call__
    result = self.forward(*input, **kwargs)
  File "/opt/conda/lib/python3.6/site-packages/torch/nn/modules/loss.py", line 932, in forward
    ignore_index=self.ignore_index, reduction=self.reduction)
  File "/opt/conda/lib/python3.6/site-packages/torch/nn/functional.py", line 2298, in cross_entropy
    return nll_loss(log_softmax(input, 1), target, weight, None, ignore_index, None, reduction)
  File "/opt/conda/lib/python3.6/site-packages/torch/nn/functional.py", line 2096, in nll_loss
    ret = torch._C._nn.nll_loss(input, target, weight, _Reduction.get_enum(reduction), ignore_index)
IndexError: Target 5 is out of bounds.
- targetはおそらくlabelのことだと思うが、0~5にしたはずなんだけどな......。
    - もしデータを全部流し見ているという話なら、5が出るまでにはしばらくかかるはずなのだが、しょっぱなからエラー出ている？のも少し謎？
    - （あんまりこんな邪道な事してる人いないやろと思って）ダメ元でエラー文で検索かけたら、割といっぱい出てきた
    - https://discuss.pytorch.org/t/indexerror-target-2-is-out-of-bounds/69614
        - torchのshapeとの不整合が原因っぽい。たぶんどっかの変数がまだ2か3のまま
        - `modeling_bert.py`のloss関数に渡してるself.num_labelsがあやしい気がする
        - BertForTokenClassification classのself.num_labelsは、config.num_labelsで定義される
            - config, modelのnum_labels確認 → configをみる限り、label2id等はちゃんと6値になっている ん？いやこれ......5値では？？？
            - num_labelsの値、6にしないといけないのでは？
            - 動いた？？？気がする？？？？？

- gpuで回す、あとその他の必要そうなoptionも
    - `--seed 0`
    - `num_train_epochs 3`
    - `--per_gpu_train_batch_size 8` :  PER_GPU_TRAIN_BATCH_SIZE Batch size per GPU/CPU for training.
    - `--per_gpu_eval_batch_size 8` : PER_GPU_EVAL_BATCH_SIZE Batch size per GPU/CPU for evaluation.
    - gpuで回すのって、もしかしてjobを投げるときに指定するのかな...？
        - とりあえずdata, outputディレクトリ以下にsampleディレクトリを作成して、`head {train/valid/test}.txt`で作成したsample dataで回してみる（これならCPUで回るでしょ）
        - 回った！！！
    - うっかりoverwriteしないように、overwriteする時もoptionで指定する様になってるのか......すごい

    - `--do_predict`（未解決）
        - つけたらエラー出た
        > Traceback (most recent call last):
        File "/uge_mnt/home/abe-k/src/transformers/examples/text-classification/run_glue.py", line 230, in <module>
            main()
        File "/uge_mnt/home/abe-k/src/transformers/examples/text-classification/run_glue.py", line 219, in main
            item = test_dataset.get_labels()[item]
        AttributeError: 'GlueDataset' object has no attribute 'get_labels'
        - `get_labels`関数は、Processorにあったような気がするが......
            - あんまりすぐ解決策が思いつかない（devの場合はどうしてるのか、とかを見ながら...って感じか？）ので、これは保留しておいて自分でpredictすることにする？
    
    - io-esaのRAIDENページを見ながら、job投げるscriptをhomeディレクトリにおく
    - `job_src`ディレクトリを作成、とりあえずそこにいっぱい書いてく？ → `dementia_run.sh`

    - 基本的にkiyonoさんのscriptに従う
        - containerが違うはずなので、そのsetupのscript名を変更
        - 仮想環境も違うはずなので、それを変更
    
    - gpuが動かない問題、[このIssue](https://github.com/huggingface/transformers/issues/2704)に従って`torch.cuda.is_available()`を試してみるとFalseが出た = torchのcudaのversionがあってない（あれ？？）
    - う〜む
        - もう一回kiyonoさんのこれでinstallしてみる
            - pip install https://download.pytorch.org/whl/cu100/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl  #CUDA10のビルド済みバイナリ
            - pip install torchvision

        - なんかpipの参照先 /opt/conda/libになってない？？
            - で、そこからuninstallしようとしてるからpermission deniedって言われてる
            - は？？？
        - export PYTHONPATH="/uge_mnt/home/abe-k/dementia_dialogue/dementia_dialogue/dementia_cuda10/lib/python3.6/site-packages/:$PYTHONPATH"
        - export LD_LIBRARY_PATH="/uge_mnt/home/abe-k/dementia_dialogue/dementia_dialogue/dementia_cuda10/lib"
            - これらを実行して`import torch`したらいつもより時間かかった
            - やったか？？ → やってない
        - じゃあこの`dementia_cuda10`は無の境地なの？./dementia_cuda10/lib以下はなんなの？？
            - `pyvenv.cfg`がconfig fileらしい、それをみると `home = /opt/local/bin`になっている
            - そうなの？？？？？？
                - 試しに`home = /uge_mnt/home/abe-k/dementia_dialogue/dementia_dialogue/dementia_cuda10/`にしてみてsource activateを実行してみるが、`which pip`しても`pipはobt/conda/bin`のまま
        - [venv documentation](https://docs.python.org/ja/3/library/venv.html)
            > 仮想環境が有効な場合 (すなわち、仮想環境の Python インタープリタを実行しているとき)、 sys.prefix と sys.exec_prefix は仮想環境のベースディレクトリを示します。 代わりに sys.base_prefix と sys.base_exec_prefix が仮想環境を作るときに使った、仮想環境ではない環境の Python がインストールされている場所を示します。 仮想環境が無効の時は、 sys.prefix は sys.base_prefix と、 sys.exec_prefix は sys.base_exec_prefix と同じになります (全て仮想環境ではない環境の Python のインストール場所を示します)。
        - sys.prefix, sys.exec_prefix, sys.base_prefix, sys.exec_prefixをprintしてみる（ついでにtorch.cuda.is_available()も）だけの`check_venv.py`を実行してみる
        - (dementia_cuda10) abe-k@~$ python check_venv.py 
        /opt/conda
        /opt/conda
        /opt/conda
        /opt/conda
        False
        - ええ.........
    
    - 新しく仮想環境「`cuda10`」を作る
        - `check_venv.py`を実行してみると、今度はしっかりとsys.prefixにcuda10が表示されている（ええ.......）
        - cuda10にtransformers, torch, torchvision, mecab-python3をinstall
        - ついでにipythonもinstall
    - **今使ってるコンテナ、cuda10.2だ.......**
        - cuda10.2はnormal torchで良いらしい
        - pip install tensorflow-gpuもする（tensorflowはtensorboardのため説が濃厚だが...）
        > #!/usr/bin/env bash
        source /fefs/opt/dgx/env_set/common_env_set.sh
        source /fefs/opt/dgx/env_set/nvcr-tensorrt-1901-py3.sh # なんかpytorch-2003-py3.shにしたらうまく行かない。これでいいならこれでいいのでは
        /usr/local/bin/nvidia_entrypoint.sh 
        - を実行すると、`torch.cuda.is_available()=True`になった！！！！！

        - jobを投げるためにshスクリプトを書いたのは良いが、なんかnvidia_entrypoint.shを実行すると一旦止まる気がする...
        - run.shが動かなくなってる..........それはそうかも........
            - ただ自分で改変したとこじゃなくて、import errorとか言う次元なんだけど.........
            - 前のtransformersに戻したらなんとかならないかな？？ -> 2.9.1？ or 2.8.0?
            - 2.9.1だと'glue_compute_metrics'がimport error, 2.8.0だとEvalPredictionがimport error
    - てか、普通にgit cloneしたtransformersからpip installしたら多分そういうエラー起きないのでは？
        - インストールしようとしたら、EnvironmentErrorが出た（ええ.......）
        - とりあえず最新版transformersをpipでinstall & git pullで最新リポジトリをinstall → してもうまくいかない
            - なんでじゃ！！！
            - ああ〜〜〜〜requirements.txtのinstallか　確かにやったわ
            - transformersをupgradeしたら、modeで指定する方に戻っている→修正
    
    - version再確認：transformers=2.10.0, cuda=10.2のtorch, torchvision, tensorflow-gpu
        - [邪道なこれ](https://qiita.com/kenta1984/items/7f3a5d859a15b20657f3)をとりあえず再現する。実際ちゃんとしたコード書く際はこのoverrideを別スクリプトに書けば良いだけと言う気もする。
            - 仮想環境内のtransformers/data内の、以下2つをいじる
                - `metrics/__init__.py` -> if elseの中にoriginal taskを追加
                - `processors/glue.py` -> OriginalProcessorを追加
    - `src/run.sh`
        - うごいた〜〜！！！しかもこの速さはGPU使ってる気がする！！ 

    - $ `qsub -cwd -jc gpu-container_g1_dev -ac d=nvcr-pytorch-2003 ~/job_src/dementia_run.sh`
    > Your job 3681867 ("dementia_run.sh") has been submitted
    - おおお！！！！？
    - $ `qstat`
    > job-ID     prior   name       user         state submit/start at     queue                          jclass                         slots ja-task-ID 
    -----------------------------------------------------------------------------------------------
    3681867 0.01000 dementia_r abe-k        r     05/23/2020 21:26:45 g1dev@dl-gpu52                 gpu-container_g1_dev.default      10        
    - おおおおお！！
    - やったらしい？？
    - まだやってないかも！？？ → outputディレクトリの中にsampleがあることによるerrorだった
    - やった！！！（output以下にcheckpointが吐き出されてる）

### 5/24(Sun)
- testsetで評価する
- その前に--do_predictしてみる？
~qsub -cwd -jc gpu-container_g1_dev -ac d=nvcr-pytorch-2003 ~/job_src/dementia_run.sh
Your job 3683056 ("dementia_run.sh") has been submitted

    - NotImplementedError
    - よくみたらget_test_examplesはdata/processors/util.pyを呼び出している
        - util.pyはいじってないので確かにエラー出る気がする
        - なんかutil.pyを覗いてみたら全部ImplementedErrorをraiseするようになっている（？？)
        - → OriginalProcessorにget_test_云々のmethodがなかったのが原因
- OriginalProcessorにget_test_examples()を実装してjob投げる

`abe-k@~$ qsub -cwd -jc gpu-container_g1_dev -ac d=nvcr-pytorch-2003 ~/job_src/dementia_run.sh`
> Your job 3683096 ("dementia_run.sh") has been submitted

- job終了
    - `output_pred/test_results_original.txt`をみてみる
    - 0か3か4しか出力していない...
        - fine-tuning足りない？
        - そもそも収束してるのか？

### 5/25(Mon)
- `./evaluate_test.py`を作成し、実行（scikit-learnのprecision, recall, f1を使うだけ）
- micro average
    > $ python ./src/predict_test.py -pred ./output_pred/test_results_original.txt -gold ./data/test.txt 
    > [INFO] 2020/05/25 AM 04:40:40 : evaluate pred_data ... 
    > precision : 0.6681187040566294
    > recall : 0.6681187040566294
    > f1_score : 0.6681187040566294

- macro average （そもそもpredictされていないlabelがあるので、zero_divisionどうこうというエラーが出てくる）
    > $ python ./src/predict_test.py -pred ./output_pred/test_results_original.txt -gold ./data/test.txt 
    > [INFO] 2020/05/25 AM 04:41:05 : evaluate pred_data ... 
    > /uge_mnt/home/abe-k/dementia_dialogue/dementia_dialogue/cuda10/lib/python3.6/site-packages/sklearn/metrics/_classification.py:1221: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
    _warn_prf(average, modifier, msg_start, len(result))
    > precision : 0.3414099972781596
    > recall : 0.3511141647754322
    > f1_score : 0.32978447158249086

- 50epoch回してみる (output dir : `output_do_pred_epoch_50`)
    - `--save_steps`を変えたい（すごい量になりそうなので）
    - 3epochで12000steps -> 50epochで200000stepts
        - とりあえず10000にしとく

- job投げる
    - `.../output/output_do_pred_epoch_50$ qsub -cwd -jc gpu-container_g1_dev -ac d=nvcr-pytorch-2003 ~/job_src/dementi a_run.sh`
        > Your job 3683907 ("dementia_run.sh") has been submitted
        - あれ、できてない
        - あっ... output directoryで実行しちゃったらそこにlogファイルが生まれるからnot emptyエラーが生じる
    - `abe-k@~$ qsub -cwd -jc gpu-container_g1_dev -ac d=nvcr-pytorch-2003 ~/job_src/dementia_run.sh`
        > Your job 3683918 ("dementia_run.sh") has been submitted

- 50は多すぎるので、10epochにしてjob投げ直す
    - $ `qsub -cwd -jc gpu-container_g1_dev -ac d=nvcr-pytorch-2003 ~/job_src/dementia_run.sh`
        > Your job 3683957 ("dementia_run.sh") has been submitted

- 10epochの結果
    - `python ./src/evaluate_test.py -pred ./output/output_do_pred_epoch_10/test_results_original.txt -gold ./data/test.txt`
        > [INFO] 2020/05/25 AM 10:07:22 : evaluate pred_data ... 
        > [INFO] 2020/05/25 AM 10:07:22 : calculate precision, recall, f1_score ... 
        precision : 0.6540520918413649
        recall : 0.6540520918413649
        f1_score : 0.6540520918413649
        > [INFO] 2020/05/25 AM 10:07:22 : make confusion_matrix ... 
        [[1357  133  590  107   18]
        [ 397  205  294   34   19]
        [ 529  114 5255  286   94]
        [ 121   28  578  289   14]
        [  55    8  362   31  101]]

        - macro平均
        precision : 0.5023437527076132
        recall : 0.42607948939354345
        f1_score : 0.4461795758548998

## TODAY
### 5/27 (Wed)
- コード整形（他のscriptからDataProcessorを継承, run_glue.pyみたいなことをする）
-  `run_classification.py`を作成
    - baseはrun_glue.py
    - 仮想環境をいじった部分をこれで補う

    - とりあえずOriginalDatasetクラスをコピーしてくる
        - ついでに変更した部分（num_labels, processor, output_mode）も変数として入れておく
    - metrics周りも修正
        - glue_compute_metrics()を参照せずに、ほぼ同じものを返すように修正
    - あとはGlueDatasetをいじればOK？
        - 結構これが調整必要
            - GlueDataTrainingArgumentsとかを参照してるが、これがtask_nameを持ってたりする
            - でもこのclassもそんなにあれじゃないか？
            - てかこのクラスいる？？

            - 新しくDataTrainingArgumentsを一応定義しておいた（task_nameに関するあれがないので、max_length程度しか働いてないが...）
        - Robertaなども、mnliなら必要だが今はいらないので一旦コメントアウトしておく

- ある程度debuggerがimport error以外吐き出さなくなった & server上のipythonで各module importもできるか確認したので、これで動くっちゃ動くはず...？

- `./job_src/dementia_run_original.sh`（job投げる用のコード）、`output_run_classification_epoch_3`（output用のディレクトリ）を作成
    - その前に一旦sampleデータで動かしてみて確認 -> 「dataclassがどうのこうの」というエラーが出る
        - `@dataclass`を忘れてた

- jobをNew Codeで投げてみる
    - `qsub -cwd -jc gpu-container_g1_dev -ac d=nvcr-pytorch-2003 ~/job_src/dementia_run.sh`

### 5/29(Fri)
- predict部を分割した`predict_test.py`とそれを実行する`run_predict_test.sh`を作成
    - ちょっとTrainer周りの仕組みをあんまり理解せずになんとかあれこれやって動いた
    - ので（？）、試しにsampleでやってみたところ`test_result.txt`の出力がフルで動かした時と異なる
        - 元は0もpredictされるが、回し直すと3しか出力しなくなる

    > 05/29/2020 03:28:53 - INFO - transformers.tokenization_utils -   Didn't find file /home/abe-k/dementia_dialogue/dementia_dialogue/output/sample_pred_2/added_tokens.json. We won't load it.
    - 標準エラー出力を見ていると、この文が悪さしてそうな気がしないでもないか？
    - いや、しかし本当にadded_tokens.jsonがないんだよな.....それってどうしようもなくない？

    > 05/29/2020 03:28:58 - INFO - transformers.trainer -   You are instantiating a Trainer but W&B is not installed. To use wandb logging, run `pip install wandb; wandb login` see https://docs.wandb.com/huggingface.
    - これも気にはなるが、たぶんこのmoduleがないのはtraining時も同じはずなので可能性は薄そう
    - やっぱりtraining時にも同じことが書いてあったので、これが原因ではなさそう

    > 05/29/2020 03:27:34 - WARNING - __main__ -   Process rank: -1, device: cuda, n_gpu: 1, distributed training: False, 16-bits training: False
    05/29/2020 03:27:34 - INFO - __main__ -   Training/evaluation parameters TrainingArguments(output_dir='/home/abe-k/dementia_dialogue/dementia_dialogue/output/sample_pred_2', overwrite_output_dir=False, do_train=True, do_eval=True, do_predict=True, evaluate_during_training=False, per_gpu_train_batch_size=8, per_gpu_eval_batch_size=8, gradient_accumulation_steps=1, learning_rate=5e-05, weight_decay=0.0, adam_epsilon=1e-08, max_grad_norm=1.0, num_train_epochs=3.0, max_steps=-1, warmup_steps=0, logging_dir=None, logging_first_step=False, logging_steps=500, save_steps=500, save_total_limit=None, no_cuda=False, seed=0, fp16=False, fp16_opt_level='O1', local_rank=-1, tpu_num_cores=None, tpu_metrics_debug=False)

    - https://github.com/huggingface/transformers/issues/375
        - 有益そうなIssue
        - ただ、やってるんだよなそれは.......
        > You can first load the original model, and then insert this line into your python file (for example, after line 607 and 610 in run_classifier.py):
        model.load_state_dict(torch.load("output_dir/pytorch_model.bin"))
        - ほ〜ん、やってみるか
        - 謎のエラーが出る
        > Traceback (most recent call last):
        File "/home/abe-k/dementia_dialogue/dementia_dialogue/src/predict_test.py", line 106, in <module>
            main()
        File "/home/abe-k/dementia_dialogue/dementia_dialogue/src/predict_test.py", line 84, in main
            compute_metrics=compute_metrics
        File "/uge_mnt/home/abe-k/dementia_dialogue/dementia_dialogue/cuda10/lib/python3.6/site-packages/transformers/trainer.py", line 190, in __init__
            self.model = model.to(args.device)
        AttributeError: '_IncompatibleKeys' object has no attribute 'to'

        - そもそもself.modelに`_IncompatibleKeys`なるものが入ってるのがだいぶ怪しいのだが......
        - https://discuss.pytorch.org/t/torch-has-not-attribute-load-state-dict/21781/10
        - 調べてたらCPU上での使い方が載ってたのでメモしとく
        - > I saved my trained Nets on GPU and now wants to use them on CPU.
            My code is:
                checkpoint = torch.load(Path1,map_location=torch.device('cpu'))
                model.load_state_dict(torch.load(Path1,map_location=torch.device(‘cpu’))[‘model_state_dict’])
                model.load_state_dict(torch.load(Path1)['model_state_dict'])
                optimizer.load_state_dict(torch.load(Path1,map_location=torch.device('cpu'))...
        
        - `model = model.load_state_dict`としているのがいけなかった（load_state_dictの返り値が_ImcompatibleKeys, ちなみにこれはmodelに対する引数の過不足を見せてくれるものっぽい）
        - 動いた！！！

- 松田さんからmodelファイルを送ってと言われたので、今のコードで10epochで学習したものをtar.gzして送る
    - jobを投げる
    - `qsub -cwd -jc gpu-container_g1_dev -ac d=nvcr-pytorch-2003 ~/job_src/dementia_run_original.sh`
    > Your job 3690325 ("dementia_run_original.sh") has been submitted

### 6/1(Mon)
- 10epoch自分のソースコードで回してみたやつ（output_run_classification_epoch_10）の結果を確認 & 松田さんにモデルファイル送る

- 結果確認
> gold ./data/test.txt
[INFO] 2020/06/01 AM 01:36:33 : evaluate pred_data ... 
[INFO] 2020/06/01 AM 01:36:33 : calculate precision, recall, f1_score ... 
precision : 0.655413376894455
recall : 0.655413376894455
f1_score : 0.655413376894455

macro: 
precision : 0.4913963281748906
recall : 0.4271724007859604
f1_score : 0.44535246297458875
[INFO] 2020/06/01 AM 01:36:34 : make confusion_matrix ... 
[[[1366  134  558  123   24]
 [ 379  210  297   45   18]
 [ 472  110 5262  317  117]
 [ 116   27  583  282   22]
 [  46   12  361   36  102]]