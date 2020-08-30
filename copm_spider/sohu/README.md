> 本文档记录了改进的基于 scrapy 的 sohu 爬虫框架

### 房产新闻

- 整体框架没有大的改变，上一版本的爬虫框架能较好重构为 scrapy 版本
- （针对数据库存储人员）由于分布式爬虫框架，各个页面访问顺序不再是上一版本中的严格顺序访问，为了保证数据的正确存储，文件目录中每个文件的文件名改为对应网页 URL 中的标识字段，从而保证区分不同页面。

### 房产问答

- 无论是问答主页还是问答内容页面，都存在动态加载问题，原本直接借助 selenium 中的 webdriver 模拟点击和滑动滚轮等操作，这里需要在中间件（middlewares.py）中实现。这里加入了三个变量来控制：
  - is_main_request：控制问答主页动态加载的中间件
  - is_detail_request：控制问答内容页面动态加载的中间件
  - is_answer_request：控制问答内容多页访问动态加载的中间件
- 根据爬虫框架的修改文件存储结构修改如下：
  - qa_dir：包含所有房产问答页面的文件夹
    - QID_dir：问题 ID 为 QID 的文件夹
      - Q_info.json：问题的详细信息
      - answer_dir：问题回复的文件夹
        - AID_info.json：每个回复的 ID 作为文件名，每个 json 文件包含一个回复
- 之所以将文件存储结构修改，一是为了充分利用 scrapy 的分布式爬取方式，但这种方法会导致存储混乱的问题；而是为了帮助后期进行页面去重爬取
- 页码访问加 ”p2“ 的方法会得到 564 的返回值？？？scrapy 会自动忽略

### 房产论坛

- 返回 None 依然会产生请求？？？

  ```python
  elif spider.name == "bbs" and request.meta["is_response_request"] == True:
      print("5: BBS response is starting...")
      driver = webdriver.Chrome()
      driver.get(request.url)
      try:
          curr_url = driver.current_url
          next_page = driver.find_element_by_link_text("下一页")
          next_page.click()
          next_url = driver.current_url
          body = driver.page_source
          driver.close()
          if next_url.split("-")[-1] != "1.html" and next_url != curr_url:
              # print(curr_url)
              # print(next_url)
              # print("======================")
              return HtmlResponse(next_url, body = body, encoding = "utf-8", request = request)    
          else:
              # print(curr_url == next_url)
              # print("++++++++++++++++++++++")
              return None
          except Exception as e:
              driver.close()
              return None
  ```

  ​