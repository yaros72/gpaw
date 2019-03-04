from myqueue.task import task


def create_tasks():
    return [
        task('gs_MoS2.py@16:25m'),
        task('gs_WSe2.py@16:25m'),
        task('bb_MoS2.py@16:20h', deps='gs_MoS2.py'),
        task('bb_WSe2.py@16:20h', deps='gs_WSe2.py'),
        task('interpolate_bb.py', deps='bb_MoS2.py,bb_WSe2.py'),
        task('interlayer.py', deps='interpolate_bb.py')]
