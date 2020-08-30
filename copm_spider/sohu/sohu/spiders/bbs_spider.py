import re
import os
import shutil
import scrapy
from sohu.items import BBSItem

bbs_city_url_list = [
    "https://bbs.focus.cn/anshan/",
    "https://bbs.focus.cn/ankang/",
    "https://bbs.focus.cn/anqing/",
    "https://bbs.focus.cn/anshun/",
    "https://bbs.focus.cn/anyang/",
    "https://bbs.focus.cn/aomen/",
    "https://bbs.focus.cn/byne/",
    "https://bbs.focus.cn/bazhong/",
    "https://bbs.focus.cn/baiyin/",
    "https://bbs.focus.cn/baise/",
    "https://bbs.focus.cn/bengbu/",
    "https://bbs.focus.cn/baotou/",
    "https://bbs.focus.cn/bd/",
    "https://bbs.focus.cn/baoshan/",
    "https://bbs.focus.cn/baoji/",
    "https://bbs.focus.cn/beihai/",
    "https://bbs.focus.cn/bj/",
    "https://bbs.focus.cn/benxi/",
    "https://bbs.focus.cn/bijie/",
    "https://bbs.focus.cn/binzhou/",
    "https://bbs.focus.cn/bozhou/",
    "https://bbs.focus.cn/cangzhou/",
    "https://bbs.focus.cn/changdu/",
    "https://bbs.focus.cn/changde/",
    "https://bbs.focus.cn/cz/",
    "https://bbs.focus.cn/cc/",
    "https://bbs.focus.cn/cs/",
    "https://bbs.focus.cn/changzhi/",
    "https://bbs.focus.cn/chaozhou/",
    "https://bbs.focus.cn/chenzhou/",
    "https://bbs.focus.cn/cd/",
    "https://bbs.focus.cn/chengde/",
    "https://bbs.focus.cn/chizhou/",
    "https://bbs.focus.cn/chifeng/",
    "https://bbs.focus.cn/chongzuo/",
    "https://bbs.focus.cn/chuzhou/",
    "https://bbs.focus.cn/cixi/",
    "https://bbs.focus.cn/cq/",
    "https://bbs.focus.cn/dazhou/",
    "https://bbs.focus.cn/dali/",
    "https://bbs.focus.cn/dl/",
    "https://bbs.focus.cn/dq/",
    "https://bbs.focus.cn/datong/",
    "https://bbs.focus.cn/dandong/",
    "https://bbs.focus.cn/deyang/",
    "https://bbs.focus.cn/dz/",
    "https://bbs.focus.cn/dingxi/",
    "https://bbs.focus.cn/dongying/",
    "https://bbs.focus.cn/dg/",
    "https://bbs.focus.cn/danzhou/",
    "https://bbs.focus.cn/erds/",
    "https://bbs.focus.cn/ezhou/",
    "https://bbs.focus.cn/enshi/",
    "https://bbs.focus.cn/fcg/",
    "https://bbs.focus.cn/fs/",
    "https://bbs.focus.cn/fz/",
    "https://bbs.focus.cn/fushun/",
    "https://bbs.focus.cn/fuzhou/",
    "https://bbs.focus.cn/fuxin/",
    "https://bbs.focus.cn/fuyang/",
    "https://bbs.focus.cn/ganzhou/",
    "https://bbs.focus.cn/guyuan/",
    "https://bbs.focus.cn/guangan/",
    "https://bbs.focus.cn/guangyuan/",
    "https://bbs.focus.cn/gz/",
    "https://bbs.focus.cn/gl/",
    "https://bbs.focus.cn/guigang/",
    "https://bbs.focus.cn/gy/",
    "https://bbs.focus.cn/hrb/",
    "https://bbs.focus.cn/haidong/",
    "https://bbs.focus.cn/hn/",
    "https://bbs.focus.cn/handan/",
    "https://bbs.focus.cn/hanzhong/",
    "https://bbs.focus.cn/hz/",
    "https://bbs.focus.cn/heze/",
    "https://bbs.focus.cn/hf/",
    "https://bbs.focus.cn/hechi/",
    "https://bbs.focus.cn/heyuan/",
    "https://bbs.focus.cn/hebi/",
    "https://bbs.focus.cn/hezhou/",
    "https://bbs.focus.cn/heihe/",
    "https://bbs.focus.cn/hs/",
    "https://bbs.focus.cn/hengyang/",
    "https://bbs.focus.cn/hhht/",
    "https://bbs.focus.cn/hlbe/",
    "https://bbs.focus.cn/huludao/",
    "https://bbs.focus.cn/huzhou/",
    "https://bbs.focus.cn/huaihua/",
    "https://bbs.focus.cn/huaian/",
    "https://bbs.focus.cn/huaibei/",
    "https://bbs.focus.cn/huainan/",
    "https://bbs.focus.cn/huanggang/",
    "https://bbs.focus.cn/huangshan/",
    "https://bbs.focus.cn/huangshi/",
    "https://bbs.focus.cn/huizhou/",
    "https://bbs.focus.cn/jixi/",
    "https://bbs.focus.cn/jian/",
    "https://bbs.focus.cn/jilin/",
    "https://bbs.focus.cn/jn/",
    "https://bbs.focus.cn/jining/",
    "https://bbs.focus.cn/jiaxing/",
    "https://bbs.focus.cn/jiayuguan/",
    "https://bbs.focus.cn/jiamusi/",
    "https://bbs.focus.cn/jiangmen/",
    "https://bbs.focus.cn/jiaozuo/",
    "https://bbs.focus.cn/jieyang/",
    "https://bbs.focus.cn/jinchang/",
    "https://bbs.focus.cn/jinhua/",
    "https://bbs.focus.cn/jinzhou/",
    "https://bbs.focus.cn/jinzhong/",
    "https://bbs.focus.cn/jingmen/",
    "https://bbs.focus.cn/jingzhou/",
    "https://bbs.focus.cn/jingdezhen/",
    "https://bbs.focus.cn/jiujiang/",
    "https://bbs.focus.cn/jiuquan/",
    "https://bbs.focus.cn/kf/",
    "https://bbs.focus.cn/km/",
    "https://bbs.focus.cn/kunshan/",
    "https://bbs.focus.cn/lasa/",
    "https://bbs.focus.cn/laiwu/",
    "https://bbs.focus.cn/laibin/",
    "https://bbs.focus.cn/lz/",
    "https://bbs.focus.cn/langfang/",
    "https://bbs.focus.cn/leshan/",
    "https://bbs.focus.cn/lijiang/",
    "https://bbs.focus.cn/ls/",
    "https://bbs.focus.cn/lyg/",
    "https://bbs.focus.cn/liaocheng/",
    "https://bbs.focus.cn/liaoyang/",
    "https://bbs.focus.cn/linzhi/",
    "https://bbs.focus.cn/lincang/",
    "https://bbs.focus.cn/linfen/",
    "https://bbs.focus.cn/linyi/",
    "https://bbs.focus.cn/liuzhou/",
    "https://bbs.focus.cn/luan/",
    "https://bbs.focus.cn/lps/",
    "https://bbs.focus.cn/longyan/",
    "https://bbs.focus.cn/longnan/",
    "https://bbs.focus.cn/loudi/",
    "https://bbs.focus.cn/luoyang/",
    "https://bbs.focus.cn/luzhou/",
    "https://bbs.focus.cn/mas/",
    "https://bbs.focus.cn/maoming/",
    "https://bbs.focus.cn/meizhou/",
    "https://bbs.focus.cn/meishan/",
    "https://bbs.focus.cn/mianyang/",
    "https://bbs.focus.cn/mdj/",
    "https://bbs.focus.cn/nc/",
    "https://bbs.focus.cn/nanchong/",
    "https://bbs.focus.cn/nj/",
    "https://bbs.focus.cn/nn/",
    "https://bbs.focus.cn/nanping/",
    "https://bbs.focus.cn/nt/",
    "https://bbs.focus.cn/nanyang/",
    "https://bbs.focus.cn/neijiang/",
    "https://bbs.focus.cn/nb/",
    "https://bbs.focus.cn/ningde/",
    "https://bbs.focus.cn/panzhihua/",
    "https://bbs.focus.cn/panjin/",
    "https://bbs.focus.cn/pds/",
    "https://bbs.focus.cn/pingliang/",
    "https://bbs.focus.cn/pt/",
    "https://bbs.focus.cn/puer/",
    "https://bbs.focus.cn/puyang/",
    "https://bbs.focus.cn/qqhe/",
    "https://bbs.focus.cn/qinzhou/",
    "https://bbs.focus.cn/qhd/",
    "https://bbs.focus.cn/qd/",
    "https://bbs.focus.cn/qingyuan/",
    "https://bbs.focus.cn/qingyang/",
    "https://bbs.focus.cn/qujing/",
    "https://bbs.focus.cn/quanzhou/",
    "https://bbs.focus.cn/quzhou/",
    "https://bbs.focus.cn/rikaze/",
    "https://bbs.focus.cn/rizhao/",
    "https://bbs.focus.cn/smx/",
    "https://bbs.focus.cn/sanming/",
    "https://bbs.focus.cn/sansha/",
    "https://bbs.focus.cn/sanya/",
    "https://bbs.focus.cn/shannan/",
    "https://bbs.focus.cn/shantou/",
    "https://bbs.focus.cn/shanwei/",
    "https://bbs.focus.cn/shangluo/",
    "https://bbs.focus.cn/shangqiu/",
    "https://bbs.focus.cn/sh/",
    "https://bbs.focus.cn/shangrao/",
    "https://bbs.focus.cn/shaoguan/",
    "https://bbs.focus.cn/shaoyang/",
    "https://bbs.focus.cn/sx/",
    "https://bbs.focus.cn/sz/",
    "https://bbs.focus.cn/sy/",
    "https://bbs.focus.cn/shiyan/",
    "https://bbs.focus.cn/sjz/",
    "https://bbs.focus.cn/shizuishan/",
    "https://bbs.focus.cn/shuangyashan/",
    "https://bbs.focus.cn/suzhou/",
    "https://bbs.focus.cn/suqian/",
    "https://bbs.focus.cn/ahsuzhou/",
    "https://bbs.focus.cn/suizhou/",
    "https://bbs.focus.cn/suihua/",
    "https://bbs.focus.cn/suining/",
    "https://bbs.focus.cn/tz/",
    "https://bbs.focus.cn/taian/",
    "https://bbs.focus.cn/jstaizhou/",
    "https://bbs.focus.cn/ty/",
    "https://bbs.focus.cn/ts/",
    "https://bbs.focus.cn/tj/",
    "https://bbs.focus.cn/tianshui/",
    "https://bbs.focus.cn/tieling/",
    "https://bbs.focus.cn/tongliao/",
    "https://bbs.focus.cn/tongchuan/",
    "https://bbs.focus.cn/tongling/",
    "https://bbs.focus.cn/tongren/",
    "https://bbs.focus.cn/luohe/",
    "https://bbs.focus.cn/weihai/",
    "https://bbs.focus.cn/weifang/",
    "https://bbs.focus.cn/weinan/",
    "https://bbs.focus.cn/wenzhou/",
    "https://bbs.focus.cn/wuhai/",
    "https://bbs.focus.cn/wlcb/",
    "https://bbs.focus.cn/wlmq/",
    "https://bbs.focus.cn/wuxi/",
    "https://bbs.focus.cn/wuhu/",
    "https://bbs.focus.cn/wuzhou/",
    "https://bbs.focus.cn/wuzhong/",
    "https://bbs.focus.cn/wh/",
    "https://bbs.focus.cn/wuwei/",
    "https://bbs.focus.cn/xian/",
    "https://bbs.focus.cn/xichang/",
    "https://bbs.focus.cn/xining/",
    "https://bbs.focus.cn/xishuangbanna/",
    "https://bbs.focus.cn/xm/",
    "https://bbs.focus.cn/xianning/",
    "https://bbs.focus.cn/xianyang/",
    "https://bbs.focus.cn/xiangyang/",
    "https://bbs.focus.cn/xiangtan/",
    "https://bbs.focus.cn/xiangxi/",
    "https://bbs.focus.cn/xiaogan/",
    "https://bbs.focus.cn/xinxiang/",
    "https://bbs.focus.cn/xinyang/",
    "https://bbs.focus.cn/xingtai/",
    "https://bbs.focus.cn/xuzhou/",
    "https://bbs.focus.cn/xuchang/",
    "https://bbs.focus.cn/xuancheng/",
    "https://bbs.focus.cn/yaan/",
    "https://bbs.focus.cn/yt/",
    "https://bbs.focus.cn/yancheng/",
    "https://bbs.focus.cn/yanan/",
    "https://bbs.focus.cn/yangzhou/",
    "https://bbs.focus.cn/yj/",
    "https://bbs.focus.cn/yibin/",
    "https://bbs.focus.cn/yichang/",
    "https://bbs.focus.cn/yichun/",
    "https://bbs.focus.cn/yiyang/",
    "https://bbs.focus.cn/yinchuan/",
    "https://bbs.focus.cn/yingtan/",
    "https://bbs.focus.cn/yingkou/",
    "https://bbs.focus.cn/yongzhou/",
    "https://bbs.focus.cn/sxyulin/",
    "https://bbs.focus.cn/yulin/",
    "https://bbs.focus.cn/yuxi/",
    "https://bbs.focus.cn/yy/",
    "https://bbs.focus.cn/yunfu/",
    "https://bbs.focus.cn/yuncheng/",
    "https://bbs.focus.cn/chaoyang/",
    "https://bbs.focus.cn/zaozhuang/",
    "https://bbs.focus.cn/zhanjiang/",
    "https://bbs.focus.cn/zhangzhou/",
    "https://bbs.focus.cn/zjj/",
    "https://bbs.focus.cn/zjk/",
    "https://bbs.focus.cn/zhangye/",
    "https://bbs.focus.cn/zhaotong/",
    "https://bbs.focus.cn/zhaoqing/",
    "https://bbs.focus.cn/zhenjiang/",
    "https://bbs.focus.cn/zz/",
    "https://bbs.focus.cn/zs/",
    "https://bbs.focus.cn/zhongwei/",
    "https://bbs.focus.cn/zhoushan/",
    "https://bbs.focus.cn/zhoukou/",
    "https://bbs.focus.cn/zh/",
    "https://bbs.focus.cn/zhuzhou/",
    "https://bbs.focus.cn/zmd/",
    "https://bbs.focus.cn/ziyang/",
    "https://bbs.focus.cn/zibo/",
    "https://bbs.focus.cn/zigong/",
    "https://bbs.focus.cn/zunyi/"
]


class BBSSpider(scrapy.Spider):
    name = "bbs"
    mediaName = '搜狐bbs'

    def __init__(self, *args, **kwargs):
        super(BBSSpider, self).__init__(*args, **kwargs)
        self.keyword = []
        with open('/var/run/copm_spider/keyword.txt', 'r') as keyword_file:
            for line in keyword_file.readlines():
                self.keyword.append(line.strip())
        
    def start_requests(self):
        for bbs_city_url in bbs_city_url_list:
            main_request = scrapy.Request(url=bbs_city_url, callback=self.parse_bbs_list)
            main_request.meta["is_main_request"] = True
            main_request.meta["is_detail_request"] = False
            main_request.meta["is_response_request"] = False
            yield main_request

    def parse_bbs_list(self, response):
        if response.status != 200:
            print("bbs list " + response.url + " is EMPTY!")
            return
        bbs_list = response.css(".user-post-h a")
        for index, qa in enumerate(bbs_list):
            curr_bbs_url = qa.xpath("./@href").extract_first()
            curr_bbs_detail_page = response.urljoin(curr_bbs_url)
            detail_request = scrapy.Request(curr_bbs_detail_page, callback=self.parse_bbs_detail)
            detail_request.meta["is_main_request"] = False
            detail_request.meta["is_detail_request"] = True
            detail_request.meta["is_response_request"] = False
            yield detail_request

    def parse_bbs_detail(self, response):
        if response.status != 200:
            print("bbs detail " + response.url + " is EMPTY!")
            return

        def extract_with_css_first(query):
            return response.css(query).xpath("string(.)").extract_first().strip()

        def extract_with_css(query):
            return response.css(query).xpath("string(.)").extract()

        page_city = response.url.split("/")[-3]
        page_id = response.url.split("/")[-1].split(".")[0]
        title = extract_with_css_first("h2")
        view_num = extract_with_css_first(".sync-click")
        response_num = re.findall(r"\d+\.?\d*", extract_with_css_first("#hd .reply-post"))[0]
        poster_id = response.css(".lz-ico+ a::attr(href)").extract_first().split("/")[-1]
        poster_level = extract_with_css_first("p.lz-score > i")
        post_time = extract_with_css_first(".header-l").split("发表于")[-1].strip()
        post_detail = extract_with_css(".luntan-post-detail")
        post_detail = ''.join(post_detail)
        post_detail = re.sub(r'</?\w+[^>]*>', '', post_detail)
        post_detail = post_detail.replace("'", '')
        post_detail = post_detail.strip()
        if view_num.find('-') != -1:
            view_num = '0'
        if response_num.find('-') != -1:
            response_num = '0'
        if poster_level.find('-') != -1:
            poster_level = '0'
        detail = {
            "url": response.url,
            "origin": "sohu",
            "title": title,
            "post_time": post_time,
            "view_num": view_num,
            "response_num": response_num,
            "poster_id": poster_id,
            "poster_level": poster_level,
            "post_detail": post_detail,
            "mediaName": self.mediaName
        }
        item = BBSItem(detail)
        if item["post_detail"] != '':
            for key in self.keyword:
                if item["post_detail"].find(key) != -1:
                    item['keyword'] = key
                    yield item
                    break
        self.log("Post %s saved" % response.url)
