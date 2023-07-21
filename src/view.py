#log
import logging
logging.basicConfig(
    level=logging.DEBUG, # ログの出力レベルを指定します。DEBUG, INFO, WARNING, ERROR, CRITICALから選択できます。
    format='%(asctime)s %(levelname)s %(message)s', # ログのフォーマットを指定します。
    datefmt='%Y-%m-%d %H:%M:%S' # ログの日付時刻フォーマットを指定します。
)

# 必要なモジュールのインポート
import collections
from detection import model # detection.py からネットワークの定義を読み込み
from flask import Flask, request, render_template, redirect
import io
from PIL import Image, ImageDraw, ImageFont
import base64
import math

from detection import model # detection.py からネットワークの定義を読み込み

'''
YOLOv8を利用して推論
'''

# YOLOv8モデルをもとに推論する
def predict(img):

    '''
    ネットワークの準備
    img : 画像データ
    conf : 確率のMIN値
    '''
    results = model(img, conf=0.6)
    # 物体名を描画する
    font = ImageFont.truetype("./ipaexg00401/ipaexg.ttf", size=60)  # フォントとサイズを指定する
    draw = ImageDraw.Draw(img)

    #class_names : 物体検出クラス
    class_names = ['柿の種', 'ピーナッツ']


    for pred in results:
        '''
        pred : tensor型
        box : 位置
        cls : 物体検出クラス
        conf : 確率
        '''
        for box, cls, conf in zip(pred.boxes.xyxy, pred.boxes.cls, pred.boxes.conf):

            '''
            バウンディングボックスを描く
            xmin : 左上
            ymin : 左下
            xmax : 右上
            ymax : 右下
            '''
            xmin, ymin, xmax, ymax = box.tolist()
            draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="red", width=5)
            
            '''
            物体名と確率を描く
            label : 物体名
            label_with_prob : 物体名と確率
            '''
            label = class_names[int(cls.numpy())]
            label_with_prob = f"{label} {conf:.2f}"
            w, h = font.getsize(label_with_prob)
            draw.rectangle([xmin, ymin, xmin+w, ymin+h], fill='red')
            draw.text((xmin, ymin), label_with_prob, fill="white", font=font)  # 物体名を描画する
    return results, img

# Flask をインスタンス化
app = Flask(__name__)

# アップロードされる拡張子の制限
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif', 'jpeg'])

#　拡張子が適切かどうかをチェック
def allwed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

'''
result.indexで表示する画像を640*640で表示するための計算
'''
def size(image):
    height, width = image.height, image.width
    magnification = height/640
    return int(height/magnification), int(width/magnification)

# URL にアクセスがあった場合の挙動の設定
@app.route('/', methods = ['GET', 'POST'])
def predicts():
    # リクエストがポストかどうかの判別
    if request.method == 'POST':
        # ファイルがなかった場合の処理
        if 'filename' not in request.files:
            return redirect(request.url)
        # データの取り出し
        file = request.files['filename']
        # ファイルのチェック
        if file and allwed_file(file.filename):

            return render_template('result.html', kakinotaneCount=1, nutsCount=2,kakinotaneRatio=50, nutsRatio=50, image='')

            #　画像ファイルに対する処理
            #　画像書き込み用バッファを確保
            buf = io.BytesIO()
            image = Image.open(file).convert('RGB')


            # 入力された画像に対して推論
            preds, draw_image = predict(image)
            pred = preds[0]
            '''
            kakinotane_count_ : 柿の種の数
            nuts_count_ : ピーナッツの数
            '''
            kakinotane_count = collections.Counter(pred.boxes.cls.numpy())
            kakinotane_count_ = kakinotane_count[0]
            nuts_count_ = kakinotane_count[1]
            
            kakinotane_list = [kakinotane_count[0], kakinotane_count[1]]
            # 最大公約数
            gcd = math.gcd(*kakinotane_list)
            kakinotane_ratio_, nuts_ratio_= (int(i / gcd) for i in kakinotane_list)
            

            '''
            result.indexで表示する画像を640*640で表示するためにリサイズ
            '''
            height, width = size(draw_image)
            draw_image = draw_image.resize((height,width))
            #　画像データをバッファに書き込む
            draw_image.save(buf, 'png')
            #　バイナリデータを base64 でエンコードして utf-8 でデコード
            base64_str = base64.b64encode(buf.getvalue()).decode('utf-8')
            #　HTML 側の src  の記述に合わせるために付帯情報付与する
            base64_data = 'data:image/png;base64,{}'.format(base64_str)
            return render_template('result.html', kakinotaneCount=kakinotane_count_, nutsCount=nuts_count_,kakinotaneRatio=kakinotane_ratio_, nutsRatio=nuts_ratio_, image=base64_data)
        
        return redirect(request.url)

    # GET メソッドの定義
    elif request.method == 'GET':
        return render_template('index.html')


# アプリケーションの実行の定義
if __name__ == '__main__':
    app.run(debug=True)