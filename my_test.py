
class cat():
    def __init__(self, new_name):
        self.name = new_name
        print("%s 来了" % self.name)

    def __del__(self):  # 内置删除函数，在被销毁之前最后一次被执行
        print("%s走了" % self.name)


tom = cat("TOM")
# __del__会在print后执行，因为tom是一个全局变量，会把整个代码执行完才会执行销毁tom
# del tom  # 在print前销毁tom
print("-" * 50)
