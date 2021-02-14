from xml.etree import ElementTree

def tokenFromXml(xmlpath:str):
    x = xmlpath#'D:/Users/poly_Z/Documents/splatmusicprj/streamingWidget/data/config.xml'               # 読み込むxmlファイルのパスを変数に記憶させる
    tree = ElementTree.parse(x)    # xmlファイルを読み込む
    root = tree.getroot()          # ルートを取得する
    # print(root.tag)                 # fruit
    ans=root.findtext('iksm_session')
    print(ans) # iksm_session
    return ans

if __name__=='__main__':
    ans=tokenFromXml('D:/Users/poly_Z/Documents/splatmusicprj/streamingWidget/data/config.xml')
    print(ans)

