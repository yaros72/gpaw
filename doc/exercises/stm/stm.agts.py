def create_tasks():
    from myqueue.task import task
    return [
        task('HAl100.py'),
        task('stm.agts.py', deps='HAl100.py')]


if __name__ == '__main__':
    import sys
    sys.argv = ['', 'HAl100.gpw']
    exec(open('stm.py').read())
