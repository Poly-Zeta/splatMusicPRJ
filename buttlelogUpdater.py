import time

import urlKnocker
import widgetWindow
import statUploader


def logUpdate(baseImg, getJsonOptionCode, delay, fontpath, iksm_cookie, stat_apikey):
    ###############################################################################
    # 戦績表示

    # 試合が終了してからAPIが応答してくれる時間まで待つ
    for i in range(int(delay)):
        time.sleep(1)
        print("waitnow "+str(i))
    # 直近50戦の簡易データを取得
    jsonData, getJsonOptionCode = urlKnocker.getJson(
        "https://app.splatoon2.nintendo.net/api/results", getJsonOptionCode
    )
    # 内容から最新の試合番号を返す
    buttleNumber, getJsonOptionCode = urlKnocker.getNewestButtleNumber(
        jsonData,
        getJsonOptionCode
    )
    # 最新の試合の詳細データを取得
    jsonData, getJsonOptionCode = urlKnocker.getJson(
        "https://app.splatoon2.nintendo.net/api/results/" +
        str(buttleNumber),
        getJsonOptionCode
    )
    # statinkへのアップロード
    statUploader.statUpload(
        jsonData,
        iksm_cookie,
        stat_apikey
    )
    # データの成形
    buttleResult, buttleRule = urlKnocker.getNewestButtleResult(
        jsonData,
        getJsonOptionCode
    )
    # データの表示
    baseImg = widgetWindow.printButtleResult(
        buttleResult,
        buttleRule,
        baseImg,
        fontpath
    )
    ###############################################################################
    return baseImg, getJsonOptionCode
