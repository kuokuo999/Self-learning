import requests
import bs4

if __name__ == '__main__':
    # 手動輸入搜尋關鍵字
    target = input("請輸入搜尋的關鍵字: ")
    
    # 手動輸入下載圖片的數量
    num_images = int(input("請輸入要下載的圖片數量: "))

    url = 'https://www.wallsauce.com/search/?search-criteria=' + target
    
    html = requests.get(url)
    web = bs4.BeautifulSoup(html.text, features='lxml')

    # 查找所有包含圖片的元素
    result = web.select(selector='.ll.mob-only.addaspect')
    
    # 計算照片數量
    total_images = len(result)
    print(f"搜尋關鍵字 '{target}' 之後，找到了 {total_images} 張照片。")

    # 根據用戶指定的數量下載圖片
    # 檔案會存在User裡面
    for i in range(min(num_images, total_images)):
        img_url = result[i]['data-src']  # 使用 'data-src' 屬性
        print(img_url)

        with open(target + str(i + 1) + '.jpg', 'wb') as file:
            picture = requests.get('https://www.wallsauce.com' + img_url)
            file.write(picture.content)
