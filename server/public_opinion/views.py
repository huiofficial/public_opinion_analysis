from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.
# 接收请求数据
def index(request):
    return render(request, 'public_opinion/index.html')
    #return HttpResponse('hello world')

def starter(request):
    return render(request, 'public_opinion/starter.html')
    #return HttpResponse('hello world')