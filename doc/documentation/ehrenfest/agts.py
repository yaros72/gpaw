from myqueue.task import task


def create_tasks():
    return [task('h2_gs.py'),
            task('h2_diss.py@8:10m', deps='h2_gs.py'),
            task('graphene_h_gs.py@8:10m'),
            task('graphene_h_prop.py@32:2h', deps='graphene_h_gs.py')]
