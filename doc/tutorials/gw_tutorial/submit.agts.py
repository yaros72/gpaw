from myqueue.task import task


def create_tasks():
    return [
        task('C_ecut_k_conv_GW.py@8:20h'),
        task('C_ecut_k_conv_plot_GW.py', deps='C_ecut_k_conv_GW.py'),
        task('C_ecut_extrap.py', deps='C_ecut_k_conv_GW.py'),
        task('C_frequency_conv.py@1:30m'),
        task('C_frequency_conv_plot.py', deps='C_frequency_conv.py'),
        task('C_equal_test.py',
            deps='C_ecut_k_conv_GW.py,C_frequency_conv.py'),
        task('BN_GW0.py@1:1h'),
        task('BN_GW0_plot.py', deps='BN_GW0.py'),
        task('MoS2_gs_GW.py@1:2h'),
        task('MoS2_GWG.py@8:20m', deps='MoS2_gs_GW.py'),
        task('MoS2_bs_plot.py', deps='MoS2_GWG.py'),
        task('check_gw.py', deps='MoS2_GWG.py')]
