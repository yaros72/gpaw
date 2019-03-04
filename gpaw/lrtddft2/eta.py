import time

class QuadraticETA:
    def __init__(self, min_times = 10):
        self.times = []
        self.min_times = max(min_times,3)

    def update(self):
        self.times.append(time.time())

    def eta(self, n):
        if ( len(self.times) < self.min_times ):
            #print '%6d' % (len(self.times))
            return 0.0

        ns = int(len(self.times) / 3.) 

        y1 = 0.0
        for i in range(len(self.times)-3*ns,len(self.times)-2*ns):
            y1 += self.times[i]
        y1 /= ns
        x1  = (len(self.times)-3*ns + len(self.times)-2*ns) / 2.

        y2 = 0.0
        for i in range(len(self.times)-2*ns,len(self.times)-ns):
            y2 += self.times[i]
        y2 /= ns
        x2  = (len(self.times)-2*ns + len(self.times)-1*ns) / 2.

        y3 = 0.0
        for i in range(len(self.times)-ns,len(self.times)):
            y3 += self.times[i]
        y3 /= ns
        x3  = (len(self.times)-1*ns + len(self.times)-0*ns) / 2.
        
        # 2nd order fit:
        #   y(x) = a * x**2 + b * x + c
        #   y'(x) = 2a * x + b
        #   y''(x) = 2a
        #   y2''(x2) = 2 a = [(y3-y2)/(x3-x2) - (y2-y1)/(x2-x1)] / [.5*(x3+x2)-.5*(x2+x1)]
        #   y2'(x2) = 2*a*x + b = (y3-y1)/(x3-x1)
        #   y2(x2) = a * x2**2 + b * x2 + c
        a = .5 * ( (y3-y2)/(x3-x2) - (y2-y1)/(x2-x1) ) / ( .5*(x3+x2)-.5*(x2+x1) )
        b = (y3-y1)/(x3-x1) - 2 * a * x2
        c = y2 - a * x2*x2 - b * x2

        #print x1,y1
        #print x2,y2
        #print x3,y3

        #print a
        #print b
        #print c

        #print '%6d %12.6lf %12.6lf %18.6lf %12.6lf %18.6lf' % (len(self.times), a, b, c-self.times[0], a * n*n + b * n + c - time.time(), a * n*n + b * n + c - self.times[0])

        return a * n*n + b * n + c - time.time()


if __name__ == "__main__":
    eta = QuadraticETA(10)

    N = 100

    print(time.time())
    t0 = time.time()
    for i in range(N):
        eta.update()
        eta.eta(N+1)
        for j in range(i+1):
            time.sleep(.001)

    t1 = time.time()
            
    print(time.time())
    
    print(t1-t0)
