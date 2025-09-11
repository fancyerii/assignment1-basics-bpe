from multiprocessing import Process, set_start_method

class Unpickleable:
    def __reduce__(self):    
        raise TypeError("I hate pickle")

    def __str__(self) -> str:
        return "Unpickleable"

def func(obj):
    print(obj)

if __name__ == '__main__':
    #set_start_method('spawn')
    set_start_method('fork')
    o = Unpickleable()
    p = Process(target=func, args=((o,)))
    p.start()
    p.join()   