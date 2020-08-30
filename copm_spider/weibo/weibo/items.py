# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class WeiboItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    #[uid, nickname, is_auth, user_url, weibo_url, content, praise_num,
    #                        retweet_num, comment_num, creat_time, all_weibo_num]
    

    weibo_id = scrapy.Field()
    weibo_url = scrapy.Field()
    weibo_content = scrapy.Field()
    weibo_repost = scrapy.Field()
    weibo_like = scrapy.Field()
    weibo_comment = scrapy.Field()
    weibo_ctime = scrapy.Field()
    
    
    user_id = scrapy.Field()
    user_name = scrapy.Field()
    user_url = scrapy.Field()
    user_verified = scrapy.Field()
    user_follow = scrapy.Field()
    user_fan = scrapy.Field()
    user_weibo = scrapy.Field()

    keyword = scrapy.Field()
    mediaName = scrapy.Field()
