1. torch环境有无
    一键执行失败
    
2.全流程。短途训练+可视化。

3.数据生成
    dpi-net 
    splish


4.数据格式
    zst 
    这个格式是不是作者写过一个接口，既然它能用很多种生成方式得到的话？
        有从numpy转换为zst的接口
    总之要么从zst之前就送入数据。要么绕过zst送入数据。
    绕过更方便。

    湍数据因为手动加载，yaml在训练脚本中不起作用了。
    yaml只是指出数据位置。




首先 待学习的数据里 涡度的效果就必须要明显才行 最好是弄些具有代表性的场景
然后学习的时候要把多个场景作为训练数据
（因为一个batch里其实包含了很多个不同的场景，应该是为了方法的泛化）
而每个场景里其实只取了3帧数据。




涡度数据是3维的。学习成本。

简单场景，水块。


数据场景+物理模拟结果。
    our_default_scene。逐帧结果在/partio里
    sim_0002只是表示一个随机种子。它最后会用在train/valid划分中。
    sim0001 和sim0002 连水块形状都不一样



压缩数据的结果。
    our_default_data里。

    sim_xxxx_split 



训练参数：
model_weights.h5

scene.json
train stretegy.yaml



【训练】
./train_network_tf.py lowfluid.yaml
    1.按原网络的方法，是在多个场景上训练少量帧。每个场景具有不同尺寸的水块和box。
        但是造随机场景不一定方便。手动造的话顶多也就造3-5个。
    
    2.在单个场景上随机截取多帧（每一帧都由pos0,pos1,pos2组成）。训练的时候会迭代1次计算loss。
        写一个程序随机截取若干个段（每个段有3帧）。
    就选湍流比较明显的那些帧数里，等间距采样段

    vel0指的是帧在solver中的速度，
    而非帧与帧之间的平均速度（因为帧与帧之间已经历经了多个step）
    
    box用法向量还是粒子
        1.如果box不是粒子而是平面，就要改网络结构。能不能用都是问题，毕竟改了结构。此外法向量的信息肯定没有粒子坐标的信息多。
        而且这样不规则的边界处理不了

        2.造一个和边界差不多大的box粒子出来。粒子距离的问题先不管了

        论文中说的是normal。
        但是输入卷积的最底层包括粒子和法向量。是否信息有些冗余？
        但还是先不改。保留网络原貌吧。

        MCVSPH好像没有提供固体粒子边界处理，是用法向量处理的。
        用之前的框架搞box。
            生成box的方法有问题。重写。

        可不可以用splishSplash搞？cconv自带的
            需要自行提供box.obj。应该是建模得来的？转换成粒子和norm确实是可以是有封装。
            其实用taichi-SPH弄个box以及法向量也没那么难，因为代码逻辑是透明的。主要是繁琐。但是上述这个有封装的代码，还需要调试。

        其实就是一个理论上的数组就可以了。不用真的去用框架造出来。用numpy造都行。


随机旋转策略。
划分valid ds。


重新设计了粒子权重。对于这种飞溅出去的粒子重要性降低
但是仍然保留了原先权重考虑表面特征的属性。
这也和核半径大小到底是多少有关。



batchSize越大，网络回传一次就要考虑越多的信息。按理来说收敛速度就会越慢。
但是同一套数据不应该重复出现在一个batch里吧？一般来说都是数据量比较大的时候分多个batch训练才对。

可能还是因为数据规模的问题。
或者完全照搬原来的网络的训练集大小以及超参数。

为啥default_data上来默认的loss就只有2？



有ckpt就会restore。（文件夹里）


【提速】
    在同一个场景中重复采样，考虑把数据保存下来节约时间  
    多开几张卡
    长时间训练必须用tmux。


【评估模型的损失】
/evaluate_network.py --trainscript train_network_tf.py --cfg default.yaml


【评估模型的预测】
./run_network.py --weights lowfluid2.h5 \
                 --scene example_scene.json \
                 --output lowfluid2_out1900 \
                 --write-ply \
                 train_network_tf.py


./run_network.py --weights lowfluid2cut.h5 \
                 --scene example_scene.json \
                 --output lowfluid2cut_out1750 \
                 --write-ply \
                 train_network_tf.py                
cp -r lowfluid2cut_out1750 

./run_network.py --weights pretrained_model_weights.h5 \
                 --scene example_scene.json \
                 --output temp_out \
                 --write-ply \
                 train_network_tf.py


训练数据：
    粒子位置，速度，（涡度）
    预测:位置矫正，速度矫正，涡度矫正。
损失函数：
    涡量损失。对于求解出的速度场求涡度。

【改进】
    领域粒子少到一定程度的时候就不要学了。权重。
    或者考虑在数据集里就去掉这部分粒子

【评价】
    如何强调涡量在这个过程中的重要性。
        可视化；
        定量指标


测试：
    1.给定无湍的模拟（只给一帧进行后续多帧的预测？任给一帧进行修复？）
    2.给定另一个场景

