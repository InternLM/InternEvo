脚本启动于 2025-04-02 17:17:20+08:00 [TERM="xterm-256color" TTY="/dev/pts/2" COLUMNS="199" LINES="53"]


Welcome to 4.19.90-2102.2.0.0066.ctl2.aarch64

System information as of time: 	2025年 04月 02日 星期三 17:17:20 CST

System load: 	[0;33;40m9.18[0m
Processes: 	2472
Memory used: 	2.5%
Swap used: 	0.0%
Usage On: 	16%
IP address: 	10.201.20.222
Users online: 	3



]0;root@node910b-0107140027:/mnt/cwai/pjlab_data/script[root@node910b-0107140027 script]# which python
/usr/bin/python
]0;root@node910b-0107140027:/mnt/cwai/pjlab_data/script[root@node910b-0107140027 script]# ls
[0m[01;32mcheck_hccl.sh[0m  [01;32mkill.sh[0m           plot_loss.py  [01;32mrun_hccl_check.sh[0m      [01;32mrun_internevo_20b_ditorch.sh[0m  [01;32mrun_interntrain_100b.sh[0m          [01;32mrun_job.sh[0m  run_temp.sh
[01;34mjin[0m            loss_en_plot.png  [01;32mrun_7b.sh[0m     [01;32mrun_internevo_100b.sh[0m  [01;32mrun_internevo_20b.sh[0m          [01;32mrun_interntrain_ditorch_100b.sh[0m  [01;32mrun.sh[0m      [01;32mstart.sh[0m
]0;root@node910b-0107140027:/mnt/cwai/pjlab_data/script[root@node910b-0107140027 script]# nano run_7b.sh
bash: nano：未找到命令
]0;root@node910b-0107140027:/mnt/cwai/pjlab_data/script[root@node910b-0107140027 script]# cd ..
]0;root@node910b-0107140027:/mnt/cwai/pjlab_data[root@node910b-0107140027 pjlab_data]# ls
[0m[01;34m100[0m           [01;34mbackup[0m            [01;34mdata[0m            [01;34minternlm2_5[0m          [01;34mjin[0m        [01;34mmain_logs_old_data[0m  [01;34mpuyu3-delivery[0m       [01;34mpuyu3-delivery_new[0m  [01;34mtmp[0m                              [01;34mvar_log_bk[0m
[01;34m1200[0m          [01;34mcode[0m              [01;34mdeeplink-rover[0m  internlm2_5-md5.txt  [01;34mlogs[0m       [01;34mmain_logs_test[0m      [01;34mpuyu3-delivery_0208[0m  [01;34mscript[0m              train-65aa8204fc2e-004705.jsonl  [01;34myuansheng[0m
[01;34mascend_image[0m  [01;34mctyun_test250107[0m  [01;34mhwtzd[0m           [01;34minternlm2_5_new[0m      [01;34mmain_logs[0m  process.sh          [01;34mpuyu3-delivery_0224[0m  [01;34mtest[0m                [01;34muser[0m                             [01;34mzos[0m
]0;root@node910b-0107140027:/mnt/cwai/pjlab_data[root@node910b-0107140027 pjlab_data]# cd data[K[K[K[Kuser
]0;root@node910b-0107140027:/mnt/cwai/pjlab_data/user[root@node910b-0107140027 user]# ls
[0m[01;34mascend[0m  [01;34mcaikun[0m  cp_nohup_job_zhumingzhu  [01;34mdongkaixing[0m  [01;34mjiaopenglong[0m  [01;34mlijiaxing[0m  [01;34mquwenwen[0m  run_intern.sh  [01;34mtangyufeng[0m  [01;34mtangzhiyi[0m  [01;34mwangqing[0m  [01;34mzhaochaoxing[0m  [01;34mzhumingzhu[0m
]0;root@node910b-0107140027:/mnt/cwai/pjlab_data/user[root@node910b-0107140027 user]# mkdir [K[K[K[K[K[Kcd quwenwen
]0;root@node910b-0107140027:/mnt/cwai/pjlab_data/user/quwenwen[root@node910b-0107140027 quwenwen]# ls
1.txt                  [0m[01;34mdarwin_ckpt[0m                      [01;34mllm_alter[0m         [01;34mps_ckpt_chunjie[0m  run_group3_zmz.sh               [01;32mrun_interntrain_100b_group01.sh[0m  [01;32mrun_ps11.sh[0m  [01;32mrun_ps5.sh[0m    test_group0.sh
2.txt                  gene_validate_data.py            [01;34mlogs[0m              rc.cfg           run_group4.sh                   [01;32mrun_interntrain_100b_group0.sh[0m   [01;32mrun_ps12.sh[0m  [01;32mrun_ps6.sh[0m    test_intern.sh
3dps.tar               [01;34minternlm_bak[0m                     [01;34mlogs_bak[0m          [01;34mRUN[0m              run_group5.sh                   [01;32mrun_interntrain_100b_group1.sh[0m   [01;32mrun_ps13.sh[0m  [01;32mrun_ps7.sh[0m    [01;34mtest_ps_ckpt[0m
[01;34mckpt[0m                   [01;34minterntrain[0m                      [01;34mlogs_old_data[0m     run_group0.sh    run_group6.sh                   [01;32mrun_interntrain_100b_no3dps.sh[0m   [01;32mrun_ps14.sh[0m  [01;32mrun_ps8.sh[0m    [01;34mtokenizes[0m
[01;34mckpt_old_data[0m          [01;34minterntrain-feat-3dps[0m            [01;34mold_codes[0m         run_group1.sh    run_group752.sh                 [01;32mrun_interntrain_100b.sh[0m          [01;32mrun_ps15.sh[0m  [01;32mrun_ps9.sh[0m    [01;32mupdate_ip.sh[0m
[01;34mckpt_test[0m              interntrain-feat-3dps.0.0.1.zip  [01;34mold_logs[0m          run_group21.sh   run_group_nops.sh               run_interntrain.sh               [01;32mrun_ps1.sh[0m   run_rover.sh  val_1500.jsonl
[01;32mconvert_100B.sh[0m        [01;34minterntrain-feat-3dps-2[0m          plot.png          run_group22.sh   run_intern.sh                   [01;32mrun_master.sh[0m                    [01;32mrun_ps2.sh[0m   [01;32mrun_test.sh[0m   [01;34mval_data[0m
cp_nohup_job           InternTrain.zip                  [01;34mps_ckpt[0m           run_group2.sh    [01;32mrun_interntrain_100b_3dps_0.sh[0m  [01;32mrun_ps0.sh[0m                       [01;32mrun_ps3.sh[0m   [01;34mtemp[0m
cp_nohup_job_8machine  [01;34mkernel_meta[0m                      [01;34mps_ckpt_8machine[0m  run_group3.sh    [01;32mrun_interntrain_100b_3dps_1.sh[0m  [01;32mrun_ps10.sh[0m                      [01;32mrun_ps4.sh[0m   temp.txt
]0;root@node910b-0107140027:/mnt/cwai/pjlab_data/user/quwenwen[root@node910b-0107140027 quwenwen]# cd ..
]0;root@node910b-0107140027:/mnt/cwai/pjlab_data/user[root@node910b-0107140027 user]# clear
[H[2J[3J]0;root@node910b-0107140027:/mnt/cwai/pjlab_data/user[root@node910b-0107140027 user]# cd [K[K[Kls
[0m[01;34mascend[0m  [01;34mcaikun[0m  cp_nohup_job_zhumingzhu  [01;34mdongkaixing[0m  [01;34mjiaopenglong[0m  [01;34mlijiaxing[0m  [01;34mquwenwen[0m  run_intern.sh  [01;34mtangyufeng[0m  [01;34mtangzhiyi[0m  [01;34mwangqing[0m  [01;34mzhaochaoxing[0m  [01;34mzhumingzhu[0m
]0;root@node910b-0107140027:/mnt/cwai/pjlab_data/user[root@node910b-0107140027 user]#  [Kcd jiaopenglong/
]0;root@node910b-0107140027:/mnt/cwai/pjlab_data/user/jiaopenglong[root@node910b-0107140027 jiaopenglong]# ls
convert2bf16_ps_ckpt.py  [0m[01;34mnew_ps_ckpt[0m                               [01;34mtmp_ckpt[0m                 transform_ckpt_0130.out  transform_ckpt_0205.out  transform_ckpt_0219.out  [01;32mtransform_ckpt_with_cast.sh[0m
data_cmp1.txt            normalized_groups_weight_init_factor.log  transform_ckpt0125.out   transform_ckpt_0131.out  transform_ckpt_0208.out  transform_ckpt_0221.out  weight_factor.log
data_cmp.txt             [01;34mrclone[0m                                    transform_ckpt_0126.out  transform_ckpt_0201.out  transform_ckpt_0210.out  transform_ckpt_0223.out
data_list1.txt           rclone.conf                               transform_ckpt_0127.out  transform_ckpt_0202.out  transform_ckpt_0212.out  transform_ckpt_0225.out
data_list.txt            rclone-v1.69.0-linux-arm64.zip            transform_ckpt_0128.out  transform_ckpt_0203.out  transform_ckpt_0216.out  transform_ckpt.out
[01;34mlog_collect_0202[0m         split_ps.py                               transform_ckpt_0129.out  transform_ckpt_0204.out  transform_ckpt_0217.out  [01;32mtransform_ckpt.sh[0m
]0;root@node910b-0107140027:/mnt/cwai/pjlab_data/user/jiaopenglong[root@node910b-0107140027 jiaopenglong]# cd ..
]0;root@node910b-0107140027:/mnt/cwai/pjlab_data/user[root@node910b-0107140027 user]# cd lijiaxing/
]0;root@node910b-0107140027:/mnt/cwai/pjlab_data/user/lijiaxing[root@node910b-0107140027 lijiaxing]# ls
debug.log  [0m[01;34mInternEvo-feat-refactor-impl[0m  InternEvo-feat-refactor-impl.zip  merge.py  [01;34mps_ckpt[0m
]0;root@node910b-0107140027:/mnt/cwai/pjlab_data/user/lijiaxing[root@node910b-0107140027 lijiaxing]# cleatr[K[Kr
[H[2J[3J]0;root@node910b-0107140027:/mnt/cwai/pjlab_data/user/lijiaxing[root@node910b-0107140027 lijiaxing]# ls[K[Kcd ..
]0;root@node910b-0107140027:/mnt/cwai/pjlab_data/user[root@node910b-0107140027 user]# ,[Kls
[0m[01;34mascend[0m  [01;34mcaikun[0m  cp_nohup_job_zhumingzhu  [01;34mdongkaixing[0m  [01;34mjiaopenglong[0m  [01;34mlijiaxing[0m  [01;34mquwenwen[0m  run_intern.sh  [01;34mtangyufeng[0m  [01;34mtangzhiyi[0m  [01;34mwangqing[0m  [01;34mzhaochaoxing[0m  [01;34mzhumingzhu[0m
]0;root@node910b-0107140027:/mnt/cwai/pjlab_data/user[root@node910b-0107140027 user]# mkdir lusitian
]0;root@node910b-0107140027:/mnt/cwai/pjlab_data/user[root@node910b-0107140027 user]# cd lusitian/
]0;root@node910b-0107140027:/mnt/cwai/pjlab_data/user/lusitian[root@node910b-0107140027 lusitian]# la
总用量 0
]0;root@node910b-0107140027:/mnt/cwai/pjlab_data/user/lusitian[root@node910b-0107140027 lusitian]# cd /
]0;root@node910b-0107140027:/[root@node910b-0107140027 /]# ls
[0m[01;36mbin[0m  [01;34mboot[0m  [01;34mdev[0m  [01;34metc[0m  [01;34mhome[0m  [01;36mlib[0m  [01;36mlib64[0m  [01;34mlog[0m  [01;34mlost+found[0m  [01;34mmedia[0m  [01;34mmnt[0m  [01;34mopt[0m  [01;34mproc[0m  [01;34mroot[0m  [01;34mrun[0m  [01;36msbin[0m  [01;34msrv[0m  [01;34msys[0m  [30;42mtmp[0m  [01;34musr[0m  [01;34mvar[0m
]0;root@node910b-0107140027:/[root@node910b-0107140027 /]# cd hi[Kome
]0;root@node910b-0107140027:/home[root@node910b-0107140027 home]# ls
[0m[01;34mHwHiAiUser[0m  [01;34mhwMindX[0m
]0;root@node910b-0107140027:/home[root@node910b-0107140027 home]# la
总用量 0
drwx------ 2 HwHiAiUser HwHiAiUser 62  5月 16  2024 [0m[01;34mHwHiAiUser[0m
drwx------ 2 hwMindX    hwMindX    62  1月  7 20:40 [01;34mhwMindX[0m
]0;root@node910b-0107140027:/home[root@node910b-0107140027 home]# cd ..
]0;root@node910b-0107140027:/[root@node910b-0107140027 /]# cd usr
]0;root@node910b-0107140027:/usr[root@node910b-0107140027 usr]# ls
[0m[01;34mbin[0m  [01;34mgames[0m  [01;34minclude[0m  [01;34mlib[0m  [01;34mlib64[0m  [01;34mlibexec[0m  [01;34mlocal[0m  [01;34mmpi[0m  [01;34msbin[0m  [01;34mshare[0m  [01;34msrc[0m  [01;36mtmp[0m
]0;root@node910b-0107140027:/usr[root@node910b-0107140027 usr]# cd share
]0;root@node910b-0107140027:/usr/share[root@node910b-0107140027 share]# ls
[0m[01;34maclocal[0m        [01;34mawk[0m              [01;34mdbus-1[0m               [01;34meula[0m        [01;34mgcc-7.3.0[0m     [01;34mgrub[0m            [01;34mibdm2.1.1[0m     [01;34mlibreport[0m  [01;36mmagic[0m         [01;34mmstflint[0m   [01;34mpolkit-1[0m       [01;34msystemtap[0m     [01;34mucx[0m
[01;34maclocal-1.16[0m   [01;34mbackgrounds[0m      [01;34mdbxtool[0m              [01;34mfactory[0m     [01;34mGConf[0m         [01;34mgtk-2.0[0m         [01;34micons[0m         [01;34mlibthai[0m    [01;34mmakedumpfile[0m  [01;34mnmap[0m       [01;34mpublicsuffix[0m   [01;34mtabset[0m        [01;34mvim[0m
[01;34mappdata[0m        [01;34mbash-completion[0m  [01;34mdefaults[0m             [01;34mfile[0m        [01;34mgdb[0m           [01;34mgtk-3.0[0m         [01;34midl[0m           [01;34mlibtool[0m    [01;34mman[0m           [01;34momf[0m        [01;34mpython-wheels[0m  [01;34mtcl8[0m          [01;34mwayland-sessions[0m
[01;34mapplications[0m   [01;34mbison[0m            [01;34mdesktop-directories[0m  [01;34mfirewalld[0m   [01;34mgettext[0m       [01;34mgtk-doc[0m         [01;34minfo[0m          [01;34mlicenses[0m   [01;34mmetainfo[0m      [01;34mos-prober[0m  [01;34mselinux[0m        [01;34mtcl8.6[0m        [01;34mX11[0m
[01;34maugeas[0m         [01;34mcmake[0m            [01;34mdict[0m                 [01;34mfish[0m        [01;34mgettext-0.21[0m  [01;34mguile[0m           [01;34mjava[0m          [01;34mlocale[0m     [01;34mmft[0m           [01;34mp11-kit[0m    [01;34mslsh[0m           [01;34mterminfo[0m      [01;34mxml[0m
[01;34mauthselect[0m     [01;32mconfig.site[0m      [01;34mdoc[0m                  [01;34mfontconfig[0m  [01;34mgir-1.0[0m       [01;34mhelp[0m            [01;34mkdump[0m         [01;34mlshw[0m       [01;34mmime[0m          [01;34mperl5[0m      [01;34msnmp[0m           [01;34mthemes[0m        [01;34mxsessions[0m
[01;34mautoconf[0m       [01;34mcracklib[0m         [01;34memacs[0m                [01;34mfonts[0m       [01;34mglib-2.0[0m      [01;34mhwdata[0m          [01;34mkeyutils[0m      [01;34mltrace[0m     [01;34mmime-info[0m     [01;34mpixmaps[0m    [01;34msounds[0m         [01;34mthumbnailers[0m  [01;34mxtables[0m
[01;34mautogen[0m        [01;34mcrypto-policies[0m  [01;34mempty[0m                [01;34mgames[0m       [01;34mgnome[0m         [01;34mi18n[0m            [01;34mlemon[0m         [01;34mlua[0m        [01;34mmisc[0m          [01;34mpkgconfig[0m  [01;34mss[0m             [01;34mtk8.6[0m         [01;34mzoneinfo[0m
[01;34mautomake-1.16[0m  [01;34mctyunos-release[0m  [01;34met[0m                   [01;36mgawk[0m        [01;34mgroff[0m         [01;34mibdiagnet2.1.1[0m  [01;34mlibgpg-error[0m  [01;34mlustre[0m     [01;34mmlnx_ofed[0m     [01;34mpki[0m        [01;34msystemd[0m        [01;34mtuned[0m         [01;34mzsh[0m
]0;root@node910b-0107140027:/usr/share[root@node910b-0107140027 share]# cd ..
]0;root@node910b-0107140027:/usr[root@node910b-0107140027 usr]# cd ..
]0;root@node910b-0107140027:/[root@node910b-0107140027 /]# ls
[0m[01;36mbin[0m  [01;34mboot[0m  [01;34mdev[0m  [01;34metc[0m  [01;34mhome[0m  [01;36mlib[0m  [01;36mlib64[0m  [01;34mlog[0m  [01;34mlost+found[0m  [01;34mmedia[0m  [01;34mmnt[0m  [01;34mopt[0m  [01;34mproc[0m  [01;34mroot[0m  [01;34mrun[0m  [01;36msbin[0m  [01;34msrv[0m  [01;34msys[0m  [30;42mtmp[0m  [01;34musr[0m  [01;34mvar[0m
]0;root@node910b-0107140027:/[root@node910b-0107140027 /]# cd mnt
]0;root@node910b-0107140027:/mnt[root@node910b-0107140027 mnt]# ls
[0m[01;34mcwai[0m  [01;34mdata01[0m  [01;34mmatrix[0m
]0;root@node910b-0107140027:/mnt[root@node910b-0107140027 mnt]# cd data01
]0;root@node910b-0107140027:/mnt/data01[root@node910b-0107140027 data01]# ls
]0;root@node910b-0107140027:/mnt/data01[root@node910b-0107140027 data01]# cd ..
]0;root@node910b-0107140027:/mnt[root@node910b-0107140027 mnt]# cd cwai
]0;root@node910b-0107140027:/mnt/cwai[root@node910b-0107140027 cwai]# ls
[0m[01;34m20250108lama70b[0m  [01;34mcaif-project-129[0m  [34;42mhw[0m  [01;34mpjlab_data[0m  [01;34mquwenwen_data[0m
]0;root@node910b-0107140027:/mnt/cwai[root@node910b-0107140027 cwai]# cd hw
]0;root@node910b-0107140027:/mnt/cwai/hw[root@node910b-0107140027 hw]# ls
[0m[01;34m8.1.RC1.B050[0m                                               [34;42mdeepseek-ai[0m       fusion_result.json  [01;34mlmdeploy[0m                                                        [01;34mmodel-weights[0m     [01;34mqwen25-7B-hf[0m
accelerate-0.26.0-py3-none-any.whl                         [01;34mdeepseek-r1-fp16[0m  [01;34mgrpo[0m                lmdeploy_deepseek.tar                                           [01;34mmpich-4.3.0rc4[0m    req.txt
antlr4-python3-runtime-4.7.2.tar.gz                        [01;34mdeepseek-r1-w8a8[0m  grpo_commod.txt     merge_weight.py                                                 [01;32mnohup.out[0m         requirements.txt
apex-0.1.dev20240909+ascend-cp310-cp310-linux_aarch64.whl  [01;34mDeepSeek-V3-w8a8[0m  hf.py               [01;32mmindie_2.0.T6-800I-A2-py3.11-openeuler24.03-lts-aarch64.tar.gz[0m  npu.sentinel.183  [01;32mtest.txt[0m
[01;32mAscend-hdk-910b-npu-driver_24.1.0.3_linux-aarch64.run[0m      [01;32mdl_w8a8.py[0m        [01;34mlib64[0m               [01;34mMindSpeed_RL_code[0m                                               npu.sentinel.78   [01;34mzjx-pip[0m
[01;32mAscend-hdk-910b-npu-firmware_7.5.0.5.220.run[0m               [34;42mdsw8a8[0m            [01;32mlib64.zip[0m           mindspeedrl.zip                                                 [01;34mpip_pkg[0m
]0;root@node910b-0107140027:/mnt/cwai/hw[root@node910b-0107140027 hw]# cd ..
]0;root@node910b-0107140027:/mnt/cwai[root@node910b-0107140027 cwai]# cd [K[K[Kls
[0m[01;34m20250108lama70b[0m  [01;34mcaif-project-129[0m  [34;42mhw[0m  [01;34mpjlab_data[0m  [01;34mquwenwen_data[0m
]0;root@node910b-0107140027:/mnt/cwai[root@node910b-0107140027 cwai]# cd pjlab_data/
]0;root@node910b-0107140027:/mnt/cwai/pjlab_data[root@node910b-0107140027 pjlab_data]# ls
[0m[01;34m100[0m           [01;34mbackup[0m            [01;34mdata[0m            [01;34minternlm2_5[0m          [01;34mjin[0m        [01;34mmain_logs_old_data[0m  [01;34mpuyu3-delivery[0m       [01;34mpuyu3-delivery_new[0m  [01;34mtmp[0m                              [01;34mvar_log_bk[0m
[01;34m1200[0m          [01;34mcode[0m              [01;34mdeeplink-rover[0m  internlm2_5-md5.txt  [01;34mlogs[0m       [01;34mmain_logs_test[0m      [01;34mpuyu3-delivery_0208[0m  [01;34mscript[0m              train-65aa8204fc2e-004705.jsonl  [01;34myuansheng[0m
[01;34mascend_image[0m  [01;34mctyun_test250107[0m  [01;34mhwtzd[0m           [01;34minternlm2_5_new[0m      [01;34mmain_logs[0m  process.sh          [01;34mpuyu3-delivery_0224[0m  [01;34mtest[0m                [01;34muser[0m                             [01;34mzos[0m
]0;root@node910b-0107140027:/mnt/cwai/pjlab_data[root@node910b-0107140027 pjlab_data]# cd ascend_image/
]0;root@node910b-0107140027:/mnt/cwai/pjlab_data/ascend_image[root@node910b-0107140027 ascend_image]# ls
3dps_torch2_1_cann_8_0_0.tar
]0;root@node910b-0107140027:/mnt/cwai/pjlab_data/ascend_image[root@node910b-0107140027 ascend_image]# cd ..
]0;root@node910b-0107140027:/mnt/cwai/pjlab_data[root@node910b-0107140027 pjlab_data]# conda
bash: conda：未找到命令
]0;root@node910b-0107140027:/mnt/cwai/pjlab_data[root@node910b-0107140027 pjlab_data]# cd[Konda list
bash: conda：未找到命令
]0;root@node910b-0107140027:/mnt/cwai/pjlab_data[root@node910b-0107140027 pjlab_data]# cd user [K/lusitian
]0;root@node910b-0107140027:/mnt/cwai/pjlab_data/user/lusitian[root@node910b-0107140027 lusitian]# ls
]0;root@node910b-0107140027:/mnt/cwai/pjlab_data/user/lusitian[root@node910b-0107140027 lusitian]# mk[K[Kcd ..
]0;root@node910b-0107140027:/mnt/cwai/pjlab_data/user[root@node910b-0107140027 user]# df -h
文件系统                                                                                           容量  已用  可用 已用% 挂载点
devtmpfs                                                                                           766G     0  766G    0% /dev
tmpfs                                                                                              766G  128K  766G    1% /dev/shm
tmpfs                                                                                              766G  4.2G  762G    1% /run
tmpfs                                                                                              766G     0  766G    0% /sys/fs/cgroup
/dev/mapper/system-lv_root                                                                         100G   16G   85G   16% /
tmpfs                                                                                              766G   64K  766G    1% /tmp
/dev/sda3                                                                                          2.0G  140M  1.8G    8% /boot
/dev/sda2                                                                                         1022M  6.5M 1016M    1% /boot/efi
/dev/nvme0n1p1                                                                                     3.0T   79G  2.9T    3% /mnt/matrix
/dev/nvme1n1p1                                                                                     3.0T   21G  2.9T    1% /mnt/data01
tmpfs                                                                                              154G     0  154G    0% /run/user/0
100.97.192.61@o2ib:100.97.192.62@o2ib:/shRoce02/c6ceb324a3d55e56e23f0dc232f3f153_zfobzkj6jl2e8dhp  1.2P  1.1P  143T   89% /mnt/cwai/pjlab_data
s3fs                                                                                                64P     0   64P    0% /mnt/cwai/quwenwen_data
tmpfs                                                                                              766G  192K  766G    1% /mnt/matrix/kubelet/pods/2ee964c0-2407-4886-bdde-f27caf627ed1/volumes/kubernetes.io~secret/nodelocaldns-token-xkf52
shm                                                                                                 64M     0   64M    0% /run/containerd/io.containerd.grpc.v1.cri/sandboxes/77acf91df09d6915c8e5b526d8a44fad3f4d14dcc55d2193e632fd294b3e7a07/shm
overlay                                                                                            3.0T   79G  2.9T    3% /run/containerd/io.containerd.runtime.v1.linux/k8s.io/77acf91df09d6915c8e5b526d8a44fad3f4d14dcc55d2193e632fd294b3e7a07/rootfs
overlay                                                                                            3.0T   79G  2.9T    3% /run/containerd/io.containerd.runtime.v1.linux/k8s.io/c26e64f9c45d2aa28cea1246f6a5c41a182ceed385bb5f562b3fb35e1811d1e8/rootfs
tmpfs                                                                                              766G  192K  766G    1% /mnt/matrix/kubelet/pods/ac1bc2dc-9d1d-4c81-b337-6e3c77f40a83/volumes/kubernetes.io~secret/calico-node-token-tkddd
tmpfs                                                                                              766G  192K  766G    1% /mnt/matrix/kubelet/pods/af77821d-68a3-4e12-87a4-5fbe771b3b96/volumes/kubernetes.io~secret/kube-proxy-token-89h5t
shm                                                                                                 64M     0   64M    0% /run/containerd/io.containerd.grpc.v1.cri/sandboxes/101c223db540e9a9eb124c3ad87355c6d7c3bcaa0d89ffe581fa13587c84cb43/shm
overlay                                                                                            3.0T   79G  2.9T    3% /run/containerd/io.containerd.runtime.v1.linux/k8s.io/101c223db540e9a9eb124c3ad87355c6d7c3bcaa0d89ffe581fa13587c84cb43/rootfs
shm                                                                                                 64M     0   64M    0% /run/containerd/io.containerd.grpc.v1.cri/sandboxes/28f64ce4e52e2c2bd768d673e61a7ff00254792109421c070e96cae9c60ebb60/shm
overlay                                                                                            3.0T   79G  2.9T    3% /run/containerd/io.containerd.runtime.v1.linux/k8s.io/28f64ce4e52e2c2bd768d673e61a7ff00254792109421c070e96cae9c60ebb60/rootfs
shm                                                                                                 64M     0   64M    0% /run/containerd/io.containerd.grpc.v1.cri/sandboxes/0cb0586ae659da6052c5ae8644acd07eb06d5651c589a96f5f25cd2046b565db/shm
overlay                                                                                            3.0T   79G  2.9T    3% /run/containerd/io.containerd.runtime.v1.linux/k8s.io/0cb0586ae659da6052c5ae8644acd07eb06d5651c589a96f5f25cd2046b565db/rootfs
overlay                                                                                            3.0T   79G  2.9T    3% /run/containerd/io.containerd.runtime.v1.linux/k8s.io/48fdd3d66d5738e83181d12125dbf8906d7d9d7ddf333a1888c653646d70ae81/rootfs
overlay                                                                                            3.0T   79G  2.9T    3% /run/containerd/io.containerd.runtime.v1.linux/k8s.io/6e501c050238f81c7804790a677f3def51c74e20ebc50c3a3f8b66e26c2fb317/rootfs
overlay                                                                                            3.0T   79G  2.9T    3% /run/containerd/io.containerd.runtime.v1.linux/k8s.io/f13ce0fa512654dbd8e8df07f7183263fca3be29ab288e49b9d0d0fe4a689121/rootfs
tmpfs                                                                                              766G  128K  766G    1% /mnt/matrix/kubelet/pods/b81c07d9-5837-42bd-854f-91886f79f83d/volumes/kubernetes.io~secret/credentials
tmpfs                                                                                              766G  192K  766G  