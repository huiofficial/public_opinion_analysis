import scrapy
import datetime
import urllib
import codecs
import re
from se.items import SogouItem
import pytz


class SogouSpider(scrapy.Spider):
    name = "sogou"
    count = 0
    mediaName = "搜狗"

    def __init__(self, page='20', *args, **kwargs):
        super(SogouSpider, self).__init__(*args, **kwargs)
        self.keyword = []
        self.page = int(page)
        self.count = 0

    @classmethod
    def _search_query(cls, query):
        """
        搜狗搜索的规则如下：
        query: 指定搜索词
        """

        url = "https://www.sogou.com/web?query=%s"
        params = query
        url = url % params

        return url

    @classmethod
    def _extract_title(cls, titles):
        if titles is None: return
        results = []
        for item in titles:
            if item is None:
                item = ''
            else:
                item = item.replace('\n', '')
                item = item[item.index('>') + 1:item.rindex('<')]
            results.append(item)
        return results

    @classmethod
    def _extract_date(cls, dates):
        if dates is None: return
        dates = [
            item.replace('\n', '')
            for item in dates
        ]
        res = []
        for date in dates:
            if date is None or date == '':
                res.append('')
                continue
            date = date.split()[-1]
            m = re.match(r'^((19|20)\d\d)-(0?[1-9]|1[012])-(0?[1-9]|[12][0-9]|3[01])$',
                         date)
            if m is not None:
                res.append(date)
            else:
                res.append(date)
        return res

    @classmethod
    def _extract_abstract(cls, abstracts):
        if abstracts is None: return
        results = []
        for item in abstracts:
            if item is None:
                item = ''
            else:
                item = item.replace('\n', '')
                item = item[item.index('>') + 1:item.rindex('<')]
            results.append(item)
        return results

    @classmethod
    def _filter_tags(cls, htmlstr):
        # 先过滤CDATA
        re_cdata = re.compile('//<!\[CDATA\[[^>]*//\]\]>', re.I)  # 匹配CDATA
        re_script = re.compile('<\s*script[^>]*>[^<]*<\s*/\s*script\s*>', re.I)  # Script
        re_style = re.compile('<\s*style[^>]*>[^<]*<\s*/\s*style\s*>', re.I)  # style
        re_br = re.compile('<br\s*?/?>')  # 处理换行
        re_h = re.compile('</?\w+[^>]*>')  # HTML标签
        re_comment = re.compile('<!--[^>]*-->')  # HTML注释
        s = re_cdata.sub('', htmlstr)  # 去掉CDATA
        s = re_script.sub('', s)  # 去掉SCRIPT
        s = re_style.sub('', s)  # 去掉style
        s = re_br.sub('\n', s)  # 将br转换为换行
        s = re_h.sub('', s)  # 去掉HTML 标签
        s = re_comment.sub('', s)  # 去掉HTML注释
        # 去掉多余的空行
        blank_line = re.compile('\n+')
        s = blank_line.sub('\n', s)
        return s

    def start_requests(self):
        # q3 = ['雄安市民服务中心', '雄安市民中心', '雄安中海物业', '雄安中海']
        # q4 = ['指数']
        # q4 = ['指数', '股份', 'A股', 'H股', '板块', '开盘', '楼市', '成交', '售楼', '公寓', '住宅', '首付', '置业', '户型', '在售', '销售', '标书', '中海油']
        self.tz = pytz.timezone('Asia/Shanghai')
        for word in self.keyword:
            query = f'"{word}"' + '+垃圾+恶心+投诉+举报+可恶+差劲+烂'
            url = SogouSpider._search_query(query=query)
            yield scrapy.Request(url=url, meta={'keyword': word}, callback=self.parse)

            # pass
            # urls = [
            #     'http://quotes.toscrape.com/page/1/',
            #     'http://quotes.toscrape.com/page/2/',
            # ]
            # for url in urls:
            #     yield scrapy.Request(url=url, callback=self.parse)

    def get_ctime(self, timestr):
        if timestr == '' or timestr == '-':
            return ''
        if timestr.find('-') != -1:
            return timestr
        else:
            # 现在时间
            d = datetime.datetime.now(self.tz)
            # N小时前
            if '小时' in timestr:
                hour = int(timestr[0:timestr.index('小时')])
                delta = datetime.timedelta(hours=hour)
                d = d - delta
            # N天前
            elif '天' in timestr:
                day = int(timestr[0:timestr.index('天')])
                delta = datetime.timedelta(days=day)
                d = d - delta
            else:
                d = d
            timestr = d.strftime('%Y-%m-%d')
        return timestr

    def parse(self, response):
        urls = response.css('h3.vrTitle').css('a::attr(href)').extract()
        titles = response.css('h3.vrTitle').css('a').extract()
        dates = response.css('div.fb').css('cite::text').extract()
        abstracts = response.css('p.str_info').extract()

        # titles = [title.replace('\n\t', '') for title in titles]
        # dates = [date.replace('\n\t', '') for date in dates]
        # abstracts = [abstract.replace('\n\t', '') for abstract in abstracts]
        titles = SogouSpider._extract_title(titles)
        titles = [
            SogouSpider._filter_tags(title)
            for title in titles
        ]
        dates = SogouSpider._extract_date(dates)
        abstracts = SogouSpider._extract_abstract(abstracts)
        abstracts = [
            SogouSpider._filter_tags(abstract)
            for abstract in abstracts
        ]

        for url, title, date, abstract in zip(urls, titles, dates, abstracts):
            item = SogouItem()
            item["url"] = "http://www.sogou.com" + url
            item["title"] = title
            item["date"] = self.get_ctime(date)
            item["abstract"] = abstract
            item["keyword"] = response.meta['keyword']
            item["mediaName"] = self.mediaName
            content = item['title'] + ' ' + item['abstract']
            if content.strip() == '' or content.find(item['keyword']) == -1:
                pass
            else:
                filtered = False
                for word in self.filterword:
                    if content.find(word) != -1:
                        filtered = True
                        break
                if not filtered:
                    yield scrapy.Request(url=item["url"], meta={"item": item}, callback=self.parse_url)

        # next page
        # next_page = response.css('div#page a::attr(href)').extract()[-1]
        next_page = response.css('a#sogou_next::attr(href)').extract_first()
        if next_page is None:
            return
        else:
            self.count += 1
            if self.count >= self.page:
                self.log("Crawled %s pages, stopped" % self.count)
                return
            else:
                yield scrapy.Request(url=response.urljoin(next_page), meta=response.meta, callback=self.parse)
                # yield scrapy.Request(url=response.urljoin(next_page), callback=self.parse)
                # self.count += 1
                # if self.count == 10:
                #     return
                # else:
                #     yield scrapy.Request(url=response.urljoin(next_page), callback=self.parse)

    def parse_url(self, response):
        item = response.meta["item"]
        item["url"] = response.url
        yield item
