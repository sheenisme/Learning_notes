## Python爬虫实战-爬取农产品数据

> 爬虫的网站：万邦国际集团。其成立于2010年，总部位于河南省郑州市，以“立足三农、保障民生、服务全国”为宗旨，业务涵盖综合性农产品冷链物流、高效生态农业开发、生鲜连锁超市、跨境电子商务、进出口贸易等农业全产业链。荣获重点龙头企业、全国农产品“综合十强市场”、“星创天地”、全国“万企帮万村”精准扶贫先进民营企业等荣誉称号。目前，集团在中牟县建设运营的万邦农产品物流园区，已累计完成投资100亿元，占地5000亩，建筑面积达350万平方米。拥有固定商户6000多家，2017年各类农副产品交易额913亿元，交易量1720万吨，位居全国前列，实现农产品“买全球、卖全国”。
>
> 其价格信息查询为get请求，网页比较规范，且短期内不会有大的变动，很容易分析，故选择之。

#### 一、使用request爬取数据

```
# _*_ coding:utf-8 _*_
# 开发人员:未央
# 开发时间:2020/4/12 16:03
# 文件名:Scrapy_lab1.py
# 开发工具:PyCharm
import csv
import codecs
import requests  # 导入requests包
from bs4 import BeautifulSoup  # 导入bs4包
from datetime import datetime


class Produce:
    price_data = []  # 农产品的价格数据列表
    item_name = ""  # 农产品的类别名

    def __init__(self, category):
        self.item_name = category
        self.price_data = []

    # 读取某一页的数据，默认是第一页
    def get_price_page_data(self, page_index=1):
        url = 'http://www.wbncp.com/PriceQuery.aspx?PageNo=' + str(
            page_index) + '&ItemName=' + self.item_name + '&DateStart=2017/10/1&DateEnd=2020/3/31 '
        strhtml = requests.get(url)  # GET方式，获取网页数据
        # print(strhtml.text)
        soup = BeautifulSoup(strhtml.text, 'html.parser')  # 解析网页文档
        # print(soup)

        table_node = soup.find_all('table')
        # number = 0
        # for table in table_node:
        #     number += 1
        #     print(number, table)
        all_price_table = table_node[21]  # 获取含有农产品价钱的table的数据
        # print(all_price_table)
        for tr in all_price_table.find_all('tr'):
            number = 0
            price_line = []
            for td in tr.find_all('td'):
                number += 1
                # print(number, td)
                if number == 1:
                    price_line.append(td.get_text().split())  # 获取品名
                elif number == 2:
                    price_line.append(td.get_text().split())  # 获取产地
                elif number == 3:
                    price_line.append(td.get_text().split())  # 获取规格
                elif number == 4:
                    price_line.append(td.get_text().split())  # 获取单位
                elif number == 5:
                    price_line.append(td.get_text().split())  # 获取最高价
                elif number == 6:
                    price_line.append(td.get_text().split())  # 获取最低价
                elif number == 7:
                    price_line.append(td.get_text().split())  # 获取均价
                elif number == 8:
                    price_line.append(datetime.strptime(td.get_text().replace('/', '-'), '%Y-%m-%d'))  # 获取日期
            self.price_data.append(price_line)
        return

    # 获取全部页面的数据
    def get_price_data(self):
        for i in range(33):
            self.get_price_page_data(str(i))
        return

    # 讲爬虫的数据写入到CSV文件，路径为：D:\Data_pytorch\名字.csv
    def data_write_csv(self):  # file_address为写入CSV文件的路径，self.price_data为要写入数据列表
        self.get_price_data()
        file_address = "D:\Data_pytorch\\" + self.item_name.__str__() + ".csv"
        file_csv = codecs.open(file_address, 'w+', 'utf-8')  # 追加
        writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        for temp_data in self.price_data:
            writer.writerow(temp_data)
        print(self.item_name + "爬虫数据保存到文件成功！")

    # 以字典类型读取csv文件,读取路径为：D:\Data_pytorch\名字.csv
    def data_reader_csv(self):
        file_address = "D:\Data_pytorch\\" + self.item_name.__str__() + ".csv"
        with open(file_address, 'r', encoding='utf8')as fp:
            # 使用列表推导式，将读取到的数据装进列表
            data_list = [i for i in csv.DictReader(fp, fieldnames=None)]  # csv.DictReader 读取到的数据是list类型
        print(self.item_name + "数据如下：")
        print(data_list)
        return data_list


list = ["白菜", "包菜", "土豆", "菠菜", "蒜苔"]
for temp_name in list:
    produce = Produce(temp_name)
    produce.data_write_csv()
    data = produce.data_reader_csv()
```

运行之后，文件显示内容如下：

![image-20200416211956378](E:\Github socures\Learning_notes\image-20200416211956378.png)

#### 二、使用Scrapy爬起数据

类似之前的学习案例，这里不再一步一步的介绍，直接上代码：

**items.py代码如下：**

```
# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html
import scrapy
from scrapy.loader import ItemLoader
from scrapy.loader.processors import TakeFirst


class PriceSpiderItemLoader(ItemLoader):
    # 自定义itemloader，用于存储爬虫所抓取的字段内容的
    default_output_processor = TakeFirst()


class PriceSpiderItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    name = scrapy.Field()  # 品名
    address = scrapy.Field()  # 产地
    norms = scrapy.Field()  # 规格
    unit = scrapy.Field()  # 单位
    high = scrapy.Field()  # 最高价
    low = scrapy.Field()  # 最低价
    price_ave = scrapy.Field()  # 均价
    price_date = scrapy.Field()  # 日期
```

**setting.py代码如下：**

```
# -*- coding: utf-8 -*-

# Scrapy settings for price_spider project
#
# For simplicity, this file contains only settings considered important or
# commonly used. You can find more settings consulting the documentation:
#
#     https://doc.scrapy.org/en/latest/topics/settings.html
#     https://doc.scrapy.org/en/latest/topics/downloader-middleware.html
#     https://doc.scrapy.org/en/latest/topics/spider-middleware.html
from scrapy.exporters import JsonLinesItemExporter


# 默认显示的中文是阅读性较差的Unicode字符
# 需要定义子类显示出原来的字符集(将父类的ensure_ascii属性设置为False即可)
class CustomJsonLinesItemExporter(JsonLinesItemExporter):
    def __init__(self, file, **kwargs):
        super(CustomJsonLinesItemExporter, self).__init__(file, ensure_ascii=False, **kwargs)


# 启用新定义的Exporter类
FEED_EXPORTERS = {
    'json': 'price_spider.settings.CustomJsonLinesItemExporter',
}

BOT_NAME = 'price_spider'

SPIDER_MODULES = ['price_spider.spiders']
NEWSPIDER_MODULE = 'price_spider.spiders'

# Crawl responsibly by identifying yourself (and your website) on the user-agent
# USER_AGENT = 'price_spider (+http://www.yourdomain.com)'

# Obey robots.txt rules
ROBOTSTXT_OBEY = False

# Configure maximum concurrent requests performed by Scrapy (default: 16)
# CONCURRENT_REQUESTS = 32

# Configure a delay for requests for the same website (default: 0)
# See https://doc.scrapy.org/en/latest/topics/settings.html#download-delay
# See also autothrottle settings and docs
DOWNLOAD_DELAY = 3
```

**爬虫逻辑（spider.py)代码如下：**

```
# _*_ coding:utf-8 _*_
# 开发人员:未央
# 开发时间:2020/4/16 14:55
# 文件名:spider.py
# 开发工具:PyCharm
import scrapy
from price_spider.items import PriceSpiderItemLoader, PriceSpiderItem


class SpiderSpider(scrapy.Spider):
    name = 'spider'
    allowed_domains = ['www.wbncp.com']
    start_urls = ['http://www.wbncp.com/PriceQuery.aspx?PageNo=1&ItemName=%e7%99%bd%e8%8f%9c&DateStart=2017/10/1'
                  '&DateEnd=2020/3/31', 'http://www.wbncp.com/PriceQuery.aspx?PageNo=1&ItemName=土豆&DateStart=2017/10/1'
                                        '&DateEnd=2020/3/31', 'http://www.wbncp.com/PriceQuery.aspx?PageNo=1&ItemName'
                                                              '=芹菜&DateStart=2017/10/1 &DateEnd=2020/3/31']

    def parse(self, response):
        item_nodes = response.xpath("//tr[@class='Center' or @class='Center Gray']")
        for item_node in item_nodes:
            item_loader = PriceSpiderItemLoader(item=PriceSpiderItem(), selector=item_node)
            item_loader.add_css("name", "td:nth-child(1) ::text")
            item_loader.add_css("address", "td:nth-child(2) ::text")
            item_loader.add_css("norms", "td:nth-child(3) ::text")
            item_loader.add_css("unit", "td:nth-child(4) ::text")
            item_loader.add_css("high", "td:nth-child(5) ::text")
            item_loader.add_css("low", "td:nth-child(6) ::text")
            item_loader.add_css("price_ave", "td:nth-child(7)::text")
            item_loader.add_css("price_date", "td:nth-child(8)::text")
            price_item = item_loader.load_item()
            yield price_item

        next_page = response.xpath("//*[@id='cphRight_lblPage']/div/a[10]/@href").extract_first()
        if next_page is not None:
            next_page = response.urljoin(next_page)
            yield scrapy.Request(next_page, callback=self.parse)
```

**替代运行命令（price_scrapy_main.py）的代码如下：**

```
# _*_ coding:utf-8 _*_
# 开发人员:未央
# 开发时间:2020/4/16 14:55
# 文件名:price_scrapy_main.py
# 开发工具:PyCharm
from scrapy.cmdline import execute

execute(["scrapy", "crawl", "spider", "-o", "price_data.csv"])
```

运作后，将csv数据导入excel中，结果如下：

![image-20200416212730748](E:\Github socures\Learning_notes\image-20200416212730748.png)

三、经验总结：

1.使用request确实比较灵活，但是如果爬取数据多很不方便，代码也会很长，还是使用scrapy方便。特别是爬取多个页面，scrapy 的横向和纵向爬取，超级腻害！

2.Scrapy主要是设置文件（setting.py）的各种设置以及爬虫文件（本文是spider.py）的爬虫逻辑，期中主要是选择器部分比较麻烦，但是通过最近的学习，基本可以通过搜索+阅读官方文档得以解决，主要是不常用可能耗时会比较长一些罢了。

3.后面好好阅读官方文档，加油！





Write by sheen