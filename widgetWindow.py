import cv2
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import time
import urlKnocker
import json

# font = ImageFont.truetype(fontpath, 30)
# fontmini = ImageFont.truetype(fontpath, 20)


def print4imgBlank(newInputTxt: str, printlog: list, base, fontpath: str):
    width, height, sep = windowSize()
    # fontpath = 'C:/Users/poly_Z/AppData/Local/Microsoft/Windows/Fonts/RocknRollOne-Regular.ttf'
    font = ImageFont.truetype(fontpath, 25)
    printlog.append(newInputTxt)
    print(printlog, len(printlog))
    while(len(printlog) > 6):
        printlog = printlog[1:]
    # outputImg=np.zeros([200,1000,3])
    outputImg = base
    cv2.rectangle(outputImg, (sep, 0), (width, 200), (0, 0, 0), thickness=-1)
    print(printlog, len(printlog))
    try:
        img_pil = Image.fromarray(outputImg)
    except:
        img_pil = Image.fromarray((outputImg * 255).astype(np.uint8))
    # img_pil = Image.fromarray(outputImg) # 配列の各値を8bit(1byte)整数型(0～255)をPIL Imageに変換。
    draw = ImageDraw.Draw(img_pil)
    for i in range(len(printlog)):
        # outputImg=cv2.putText(outputImg,str(printlog[i]),(0,(i+1)*30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
        # draw.text((910,(i+1)*30-20),str(printlog[i]), font = font , fill = (255,255,255,0))
        draw.text((sep+10, i*30),
                  str(printlog[i]), font=font, fill=(255, 255, 255, 0))
    outputImg = np.array(img_pil)
    cv2.imshow("frame", outputImg)
    return printlog, outputImg


def printButtleResult(result: dict, rule: int, base, fontpath: str):
    width, height, sep = windowSize()
    # fontpath ='C:/Users/poly_Z/AppData/Local/Microsoft/Windows/Fonts/cinecaption226.ttf'
    # fontpath ='C:/Users/poly_Z/AppData/Local/Microsoft/Windows/Fonts/x8y12pxTheStrongGamer.ttf'
    # fontpath ='C:/Users/poly_Z/AppData/Local/Microsoft/Windows/Fonts/memoir-square.otf'

    # fontpath = 'C:/Users/poly_Z/AppData/Local/Microsoft/Windows/Fonts/RocknRollOne-Regular.ttf'
    fontmini = ImageFont.truetype(fontpath, 20)
    fontmiddle = ImageFont.truetype(fontpath, 24)
    fontbig = ImageFont.truetype(fontpath, 32)
    print(rule)
    outputImg = base
    cv2.rectangle(outputImg, (0, 0), (sep-1, height),
                  (30, 30, 30), thickness=-1)
    cv2.line(outputImg, (0, 85), (sep-1, 85), (130, 130, 130), thickness=2)
    cv2.line(outputImg, (0, 116), (sep-1, 116), (130, 130, 130), thickness=1)
    cv2.line(outputImg, (0, 142), (sep-1, 142), (130, 130, 130), thickness=1)
    cv2.line(outputImg, (0, 168), (sep-1, 168), (130, 130, 130), thickness=1)
    try:
        img_pil = Image.fromarray(outputImg)
    except:
        img_pil = Image.fromarray((outputImg * 255).astype(np.uint8))
    draw = ImageDraw.Draw(img_pil)
    if(rule == -1):
        pass
    elif(rule == 0):
        drawRuleAndStage(draw, result, fontbig)
        drawCount(draw, result, fontbig, fontmiddle)
        drawLP(draw, result, fontmiddle)
        drawButtleScore(draw, result, fontmini)
    elif(rule == 1):
        drawRuleAndStage(draw, result, fontbig)
        drawCount(draw, result, fontbig, fontmiddle)
        drawButtleScore(draw, result, fontmini)
    elif(rule == 2 or rule == 3):
        drawRuleAndStage(draw, result, fontbig)
        drawPaint(draw, result, fontbig, fontmiddle)
        drawButtleScore(draw, result, fontmini)
    elif(rule == 4):
        drawRuleAndStage(draw, result, fontbig)
        drawCount(draw, result, fontbig, fontmiddle)
        drawXP(draw, result, fontmiddle)
        drawButtleScore(draw, result, fontmini)
    outputImg = np.array(img_pil)
    cv2.imshow("frame", outputImg)
    return outputImg


def drawRuleAndStage(draw, result, fontbig):
    draw.text((3, 3), result["rule"]+" "+result["stage"],
              font=fontbig, fill=(255, 255, 255, 0))
    # draw.text((3,3),result["rule"]+" "+result["stage"], font = fontbig , fill = (255,255,255,0))


def drawCount(draw, result, fontbig, fontmiddle):
    draw.text((700, 3), result["vicordef"], font=fontbig,
              fill=vicOrDefColor(result["vicordef"]))
    draw.text((700, 38), str(result["ourcount"])+" - "+str(
        result["enemycount"]), font=fontmiddle, fill=(255, 255, 255, 0))


def drawPaint(draw, result, fontbig, fontmiddle):
    draw.text(
        (700, 3),
        result["vicordef"],
        font=fontbig,
        fill=vicOrDefColor(result["vicordef"])
    )
    draw.text((700, 38), str(result["ourpaint"])+"% - "+str(
        result["enemypaint"])+"%", font=fontmiddle, fill=(255, 255, 255, 0))


def vicOrDefColor(ans):
    if(ans == "WIN!"):
        return (127,  0, 228, 0)
    elif(ans == "LOSE…"):
        return (34, 228,  0, 0)
    else:
        return (255, 255, 255, 0)


def drawLP(draw, result, fontmiddle):
    draw.text((3, 50), "MAX LP:"+str(result["ourmaxlp"])+"   NOW LP:"+str(
        result["ourlp"])+"   ENEMY LP:"+str(result["enemylp"]), font=fontmiddle, fill=(150, 150, 150, 0))


def drawXP(draw, result, fontmiddle):
    draw.text((3, 50), "NOW XP:"+str(result["xp"])+"   ESTIMATE XP:"+str(
        result["estimateXp"]), font=fontmiddle, fill=(150, 150, 150, 0))


def drawButtleScore(draw, result, fontmini):
    width, height, sep = windowSize()
    ybase = 90
    # shift=215
    shift = 205
    for i in range(4):
        drawOurButtleScore(draw, result, fontmini, 3, ybase+i*26, shift, i)
    i = 0
    for i in range(4):
        drawEnemyButtleScore(draw, result, fontmini, sep //
                             2+10, ybase+i*26, shift, i)


def drawOurButtleScore(draw, result, fontmini, posX, posY, shift, playerNum):
    draw.text((posX, posY), result["ourlog"][playerNum]
              ["name"], font=fontmini, fill=(255, 255, 255, 0))
    draw.text((posX+shift, posY), str(result["ourlog"][playerNum]["kill"]).rjust(2)+"k "+str(result["ourlog"][playerNum]["assist"]).rjust(2)+"a "+str(result["ourlog"][playerNum]["death"]).rjust(
        2)+"d "+str(result["ourlog"][playerNum]["sp"]).rjust(2)+"s "+str(result["ourlog"][playerNum]["paint"]).rjust(4)+"p", font=fontmini, fill=(255, 255, 255, 0))


def drawEnemyButtleScore(draw, result, fontmini, posX, posY, shift, playerNum):
    draw.text((posX, posY), result["enemylog"][playerNum]
              ["name"], font=fontmini, fill=(255, 255, 255, 0))
    draw.text((posX+shift, posY), str(result["enemylog"][playerNum]["kill"]).rjust(2)+"k "+str(result["enemylog"][playerNum]["assist"]).rjust(2)+"a "+str(result["enemylog"][playerNum]["death"]).rjust(
        2)+"d "+str(result["enemylog"][playerNum]["sp"]).rjust(2)+"s "+str(result["enemylog"][playerNum]["paint"]).rjust(4)+"p", font=fontmini, fill=(255, 255, 255, 0))


def windowSize():
    width = 1500
    height = 200
    sep = 1140
    return width, height, sep


if __name__ == "__main__":
    filepath_f = open('./filepath.json', 'r')
    filepath_json_dict = json.load(filepath_f)
    viewerFontPath = filepath_json_dict["rocknrollone_font"]
    blankLogtxt = []
    wid, hei, sep = windowSize()
    baseImg = np.zeros([hei, wid, 3])
    for i in range(1):
        blankLogtxt, baseImg = print4imgBlank(
            "LABEL:endolphinsurge", blankLogtxt, baseImg, viewerFontPath)
        cv2.waitKey(0)
    for i in range(1):
        dic, rule = urlKnocker.getNewestButtleResult(
            urlKnocker.readTemplateJson(), 200)
        baseImg = printButtleResult(dic, rule, baseImg, viewerFontPath)
        cv2.waitKey(0)
