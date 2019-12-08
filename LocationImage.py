import exifread
import json
import urllib.request

# Open image file for reading (binary mode)
def  gettags():
    f = open('001.jpg', 'rb')
    # Return Exif tags
    tags = exifread.process_file(f)
    '''
    #打印所有照片信息
    for tag in tags.keys():
        print("Key: {}, value {}".format(tag, tags[tag]))
    '''

    # 打印照片其中一些信息
    print('拍摄时间：', tags['EXIF DateTimeOriginal'])
    print('照相机制造商：', tags['Image Make'])
    print('照相机型号：', tags['Image Model'])
    print('照片尺寸：', tags['EXIF ExifImageWidth'], tags['EXIF ExifImageLength'])
    return  tags


# 获取经度或纬度
def getLatOrLng(refKey, tudeKey, tags):
    if refKey not in tags:
        return None
    ref = tags[refKey].printable
    # LatOrLng = tags[tudeKey].printable[1:-1].replace(" ", "").replace("/", ",").split(",")
    LatOrLng = tags[tudeKey].printable[1:-1].replace(" ", "").replace("/", ",").split(",")
    while (len(LatOrLng) < 4):
        LatOrLng.append(1.0)
    if (len(LatOrLng) > 4):
        LatOrLng = LatOrLng[:4]

    LatOrLng = float(LatOrLng[0]) + float(LatOrLng[1]) / 60 + float(LatOrLng[2]) / float(LatOrLng[3]) / 3600

    if refKey == 'GPS GPSLatitudeRef' and tags[refKey].printable != "N": # 获取纬度
        LatOrLng = LatOrLng * (-1)
    if refKey == 'GPS GPSLongitudeRef' and tags[refKey].printable != "E": # 获取经度
        LatOrLng = LatOrLng * (-1)

    return LatOrLng

# 调用百度地图API通过经纬度获取位置
def getlocation(lat, lng):
    url = "https://restapi.amap.com/v3/geocode/regeo?output=xml&location="+ lat + ',' + lng + "&key=<用户的key>&radius=1000&extensions=all"
    # url = 'http://api.map.baidu.com/geocoder/v2/?location=' + lat + ',' + lng + '&output=json&pois=1&ak=申请的百度地图KEY'
    req = urllib.request.urlopen("https://www.baidu.com/")
    res = req.read().decode("utf-8")
    str = json.loads(res)
    # print(str)
    jsonResult = str.get('result')
    formatted_address = jsonResult.get('formatted_address')
    return formatted_address

def getLLatAndLng(tags):
    lat = getLatOrLng('GPS GPSLatitudeRef', 'GPS GPSLatitude', tags)  # 纬度
    lng = getLatOrLng('GPS GPSLongitudeRef', 'GPS GPSLongitude', tags) # 经度

    return lat, lng


def Location(lat, lng):
    url ="https://apis.map.qq.com/ws/geocoder/v1/?location=" + lat +"," +lng +"&key=M22BZ-WIZCG-7YXQ3-I6WYL-VSTTV-ZKFHG&get_poi=1"
    req = urllib.request.urlopen(url)
    res = req.read().decode("utf-8")
    str = json.loads(res)
    # print(str)
    print("位置信息:")

    print("位置一: ",str['result']['address'])
    print("位置二: ",str['result']['formatted_addresses']['recommend'])
    print("位置三: ",str['result']['formatted_addresses']['rough'])

def main():
    tags = gettags()
    lat, lng = getLLatAndLng(tags)
    if lat == None or lng == None or (lat == 0.0 and lng == 0.0):
        print("获取经纬度信息失败")
        return
    print('纬度:{}\n经度：{}'.format(lat, lng))
    Location(str(lat), str(lng))


#我的开发密钥： kye=M22BZ-WIZCG-7YXQ3-I6WYL-VSTTV-ZKFHG

if __name__ == '__main__':
   main()

