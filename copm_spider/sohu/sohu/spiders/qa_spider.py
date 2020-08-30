import re
import os
import shutil
import scrapy
from sohu.items import QAItem
import json


class QASpider(scrapy.Spider):
    name = "qa"
    mediaName = "搜狐问答"
    
    def __init__(self, *args, **kwargs):
        super(QASpider, self).__init__(*args, **kwargs)
        self.keyword = []
        with open('/var/run/copm_spider/keyword.txt', 'r') as keyword_file:
            for line in keyword_file.readlines():
                self.keyword.append(line.strip())
 
    def start_requests(self):
        url = "https://ask.focus.cn/"
        main_request = scrapy.Request(url=url, callback=self.parse_qa_list)
        main_request.meta["is_main_request"] = True
        main_request.meta["is_detail_request"] = False
        main_request.meta["is_answer_request"] = False
        yield main_request

    def parse_qa_list(self, response):
        if response.status != 200:
            self.log("qa list " + response.url + " is EMPTY!")
            return
        qa_list = response.css("div.cell > div.inner > a.question-info")
        for index, qa in enumerate(qa_list):
            curr_qa_url = "https://ask.focus.cn" + qa.xpath("./@href").extract_first()
            curr_qa_detail_page = response.urljoin(curr_qa_url)
            detail_request = scrapy.Request(curr_qa_detail_page, callback=self.parse_qa_detail)
            detail_request.meta["is_detail_request"] = True
            detail_request.meta["is_main_request"] = False
            detail_request.meta["is_answer_request"] = False
            yield detail_request

    def parse_qa_detail(self, response):
        if response.status != 200:
            self.log("qa detail " + response.url + " is EMPTY!")
            return

        def extract_with_css_first(query):
            return response.css(query).xpath("string(.)").extract_first().strip()

        def extract_with_css(query):
            return response.css(query).xpath("string(.)").extract()

        page_id = response.url.split("/")[-2]
        question_title = extract_with_css_first(".question-title")
        question_time = extract_with_css_first(".quesion-time")
        question_detail = re.sub(r'</?\w+[^>]*>', '', extract_with_css_first(".question-richText"))
        question_tag_list = [question_tag.strip() for question_tag in extract_with_css(".tag a")]
        question_tag_list = list(set(question_tag_list))
        question_follow_num = extract_with_css_first(".quesfollownum")
        question_answer_num = extract_with_css_first(".answer-count")
        questioner_id = \
            response.css("div.module-questioner > div.questioner-info > a").xpath("./@href").extract_first().split("/")[
                -1]
        detail = {
            "url": response.url,
            "origin": "sohu",
            "title": question_title,
            "question_time": question_time,
            "question_detail": question_detail,
            "question_tags": question_tag_list,
            "question_follow_num": int(question_follow_num),
            "question_answer_num": int(question_answer_num),
            "mediaName": self.mediaName,
        }
        item = QAItem(detail)
        search_str = item["title"] + " " + item["question_detail"]
        if search_str != " ":
            for key in self.keyword:
                if search_str.find(key) != -1:
                    item["keyword"] = key
                    yield item
                    break
        self.log("Question %s saved" % response.url)
