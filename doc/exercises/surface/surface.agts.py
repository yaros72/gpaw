def create_tasks():
    from myqueue.task import task
    return [
        task('surface.agts.py'),
        task('work_function.py', deps='surface.agts.py')]


if __name__ == '__main__':
    exec(open('Al100.py').read(), {'k': 6, 'N': 5})
