import scrapy
import datetime
import urllib
import codecs
import re
from se.items import BaiduItem
import pytz


class BaiduSpider(scrapy.Spider):
    name = "baidu"
    count = 0
    mediaName = '百度'

    def __init__(self, page='20', *args, **kwargs):
        super(BaiduSpider, self).__init__(*args, **kwargs)
        self.page = int(page)
        self.count = 0
        
    @classmethod
    def _search_query(cls, q3, q1=[], q2=[], q4=[], q5='', q6='', ft='', gpc=''):
        """
        百度搜索的规则如下：
        q1: 包含以下全部的关键词
        q2: 包含以下的完整关键词
        q3: 包含以下任意一个关键词
        q4: 不包含以下关键词
        q5: 关键词位置
        q6: 限定要搜索的网站
        ft: 文档格式
        gpc: 限定要搜索的网页的时间
        """
        assert isinstance(q1, list), 'Invalid type `q1`({})'.format(type(q1))
        assert isinstance(q2, list), 'Invalid type `q2`({})'.format(type(q2))
        assert isinstance(q3, list), 'Invalid type `q3`({})'.format(type(q3))
        assert isinstance(q4, list), 'Invalid type `q4`({})'.format(type(q4))
        assert q5 in {'', '1', '2'}, 'Invalid params `q5`({})'.format(q5)
        assert ft in {'', 'pdf', 'doc', 'xls', 'ppt', 'rtf', 'all'}, 'Invalid params `ft`({})'.format(ft)
        assert gpc in {'', '1day', '1week', '1month', '1year'}, 'Invalid params `gpc`({})'.format(gpc)

        if gpc == '':
            gpc = 'stf'
        else:
            gpc = 'stf={start},{end}|stftype=1'
            end = datetime.datetime.now()
            if gpc == '1day':
                delta = datetime.timedelta(days=1)
            elif gpc == '1week':
                delta = datetime.timedelta(days=7)
            elif gpc == '1month':
                delta = datetime.timedelta(days=31)
            elif gpc == '1year':
                delta = datetime.timedelta(days=365)
            else:
                assert False, "cannot reach here."
            start = end - delta
            gpc.format(start=str(start.timestamp()), end=str(end.timestamp))

        q1 = ' '.join(q1)
        q2 = ' '.join(q2)
        q3 = '+'.join(q3)
        q4 = ' '.join(q4)
        kv = {
            'q1': q1,
            'q2': q2,
            'q3': q3,
            'q4': q4,
            'q5': q5,
            'q6': q6,
            'ft': ft,
            'gpc': gpc
        }
        # print(kv)

        url = "http://www.baidu.com/s?%s"
        params = urllib.parse.urlencode(kv)
        url = url % params
        url = url.replace('%2B', '+')

        return url

    @classmethod
    def _extract_title(cls, titles):
        if titles is None: return
        results = []
        for item in titles:
            if item is None:
                item = ''
            else:
                item = item.replace('\n\t', '')
                m = re.search(r'target="_blank">(.*)</a>', item.replace('\n\t', ''))
                if m is None:
                    item = ''
                else:
                    item = m.group(1)
            results.append(item)
        return results

    @classmethod
    def _extract_date(cls, dates):
        if dates is None: return
        return [
            item.replace('\n\t', '')
            for item in dates
        ]

    @classmethod
    def _extract_abstract(cls, abstracts):
        if abstracts is None: return
        results = []
        for item in abstracts:
            if item is None:
                item = ''
            else:
                item = item.replace('\n\t', '')
                m = re.search(r'</span>(.*)</div>', item.replace('\n\t', ''))
                if m is None:
                    item = ''
                else:
                    item = m.group(1)
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
        self.tz = pytz.timezone('Asia/Shanghai')
        # # q4 = ['指数', '股份', 'A股', 'H股', '板块', '开盘', '楼市', '成交', '售楼', '公寓', '住宅', '首付', '置业', '户型', '在售', '销售', '标书', '中海油']
        emotion_word = ['垃圾', '恶心', '投诉', '可恶', '差劲', '烂', '举报']
        for key in self.keyword:
            url = BaiduSpider._search_query(q2=[key], q3=emotion_word)
            yield scrapy.Request(url=url, meta={'keyword': key}, callback=self.parse)

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
        if timestr.find('年') != -1 and timestr.find('月') != -1 and timestr.find('日') != -1:
            timestr = re.sub('\D+', '-', timestr)[:-1]
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
        urls = response.css('div.result').css('h3.t').css('a::attr(href)').extract()
        titles = response.css('div.result').css('h3.t').css('a').extract()
        dates = response.css('span.newTimeFactor_before_abs::text').extract()
        abstracts = response.css('div.c-abstract').extract()

        # titles = [title.replace('\n\t', '') for title in titles]
        # dates = [date.replace('\n\t', '') for date in dates]
        # abstracts = [abstract.replace('\n\t', '') for abstract in abstracts]
        titles = BaiduSpider._extract_title(titles)
        titles = [
            BaiduSpider._filter_tags(title)
            for title in titles
        ]
        dates = BaiduSpider._extract_date(dates)
        abstracts = BaiduSpider._extract_abstract(abstracts)
        abstracts = [
            BaiduSpider._filter_tags(abstract)
            for abstract in abstracts
        ]

        for url, title, date, abstract in zip(urls, titles, dates, abstracts):
            item = BaiduItem()
            item["url"] = url
            item["title"] = title
            item["date"] = self.get_ctime(date.replace('\xa0-\xa0', ''))
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
        # next_page = response.css('.n::attr(href)').extract_first()
        next_page = response.css('div#page a::attr(href)').extract()[-1]
        if next_page is None:
            return
        else:
            self.count += 1
            if self.count >= self.page:
                self.log("Crawled %s pages, stopped" % self.count)
                return
            else:
                yield scrapy.Request(url=response.urljoin(next_page), meta=response.meta, callback=self.parse)
                # self.count += 1
                # if self.count == 10:
                #     return
                # else:
                #     yield scrapy.Request(url=response.urljoin(next_page), callback=self.parse)

    def parse_url(self, response):
        item = response.meta["item"]
        item["url"] = response.url
        yield item
