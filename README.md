# incremental-learning-world-record-mnist-fashion-v4

accuracy score from 98.0 ~ 99.*

key ideas: 
   target/Y-decomposize by difficultly, most of subtask can archive 99.9+.
   subnetwork picket, re-seed
   subnetwork promote independently.
   you can think this is a specilal type of ensemble, but we still think there are many difference.
   the total parameters is NOT a problem, you can combin or orgnize them with any trick.

Let me explanation: (English & Chinese/中文)
1. this dataset and task, there are 10 classes.
   我们这个数据集和任务，有10个类需要去分类。
2. there is a better network, can get 96.7%; in fact, we can use more tricks to get 97.0%.  not very difficult.
   已经有一个（些）好的网络，能够达到96.7（参见数据集官网，我们前面提交的网络），事实上，我们能够试探一些其他的技巧达到97.%，不是特别难。
3. later, let us try our ideas：
   接下来，让我们试试我们的做法：
4. decompose the classifition task from one-network-ten-classes to n(a1+a(n-2))/2=10(1+8)/2=45 one-one-classification-nerworks.
   分解分类任务，从一个大网络搞定10类，到“等差数据公式”算一下需要的两两PK的小网络组合。
   before: 以前
   X --> One-Network --> Y(0...9)
   now: 现在
   X --> 45-Difference-Mini-Network --> M((0,1),(0,2),...(1,2),...(8,9))  --> Vote-Routing/Or/Emsemble-Network  --> Y
5. how to get better score: 如何刷成绩？
   subnet: 0-1:   achieve, > 99.*
   ......  *-*:   achieve, > 99.*
   subnet: 2-6    difficult,  >96.7  (one-one-pk, is very easy vs. big-network, any trick you can use!) 一对一PK比一网络搞定全分类简单多了
   ......  *-*:   achieve, > 99.*
   subnet: 4-6    difficult,  >96.7  (one-one-pk, is very easy vs. big-network, any trick you can use!)
   ......  *-*:   achieve, > 99.*
   subnet: 8-9:   achieve, > 99.*
   
   Note: for A-pk-B network, if you give a C, almost it will give you a random result between A and B. It is OK for later voting.
   注意：对于一个只认识A，B的网络，如果你给一个C，基本上会近似随机挑选的方式从A,B中选择,这是对后面的投票阶段来说，我不认识我就随机选。
6. there answer is: 答案是：
   give a test-sample,such as 8:
   假设给一个测试样本比如8:
   there are 9 subnetworks trained by *-9 will vote: (0,8)...(8,9), and, most of them will vote 8 with high-probability and has generalization-baseline.  
   其中9个与于8区分的小网络，他们大部分都会高概率的投出8，这个泛化是由底线的。
   there are rest 45-9 subnetworks will not vote 8 with high-probability.
   其他45-9个网络与8无关的，基本也不会投出8来。
7. so the final result will be better.
   所以，最终结果比较好。
   


one of our ideas is: if the total task is difficult, we can use some simple sub-network to do, any combinition is OK(one-one, tree, etc.) we think human can use this ability, one of our brain's abilities is using neurals to flexible-organize under some biger framwork, not today's NN paradigm: overemphasize end-2-end in a whole network.

if you guys understand our ideas, you can see, most of classes are SIMPLE to classify, we can use above we mentioned simplest FULL-CONNECTION to classify and get almost 99.9...9%. we can just focus on these classes PK(from difficult to simple) use ‘DIFFERENCE NN-Architecute，DIFFERENCE NN-Instances, DIFFERENCE NN-Instance-with RANDOM-SEED’ (This seed is MOST-IMPORTANT, not a joke.)：
4-6
2-6 4-6
0-6

In Chinese: Why DIFFERENCE *** are most important? （if somebody has free-time, can translate it into English.）

人的大脑高度并行，高度步骤与步骤直接解耦。
把问题想象成10*10的PK。
绝大部分类的PK是极其简单：几乎百分百。 好比区分你爸爸和你妈妈一样有信心: 100%。
比较难的是，类似本数据集，4-6(the first class is 0)，有些衣服的区分，人几乎都不知道是个啥规则。（这也是这个数据集的牛逼或者不好的地方；牛逼就是要挑战你的算法，不好的地方我感觉和对随机化的东西做分类一样没有太大意义，只是考验了NN的拟合和记忆了能力。打比赛到最后，就看谁随机数牛逼/运气好，那个随机数恰好有比较好的对test-dataset的泛化能力。如果真正和的研究者，研究过4-6这种类的PK，就知道的我说的啥意思了。所以，我们理解，这个问题，99.9%就是上界了，好比mnist的99.97%）
用不同的搞法，去搞最难的几类，然后组合起来的network'SSS'，就和我们人一样有信心。这个难的类的处理，好比我们对男女一般都能从”高阶 or 简答 or 基本 or 通用“的特征上区分，对于同卵双胞胎，可能要关注眉间某个小差别点。
6 我们AGI系统的一部分，就是networks-dispatching，对分类问题，networks-dispatching是可用能力之一。
these ideas are NOT this dataset-specific, are NOT this task-specific.
we can use human-readable code like we submited or NN code like our brain created by GOD/Demiurge.
In Chinese:
这些搞法，并不是这个只适用于这个数据集，或者这个任务；这是一个通用的做法。
至于如何实现，可以用我们提交的代码展示的方法，这个一般程序员都很容易看懂都理解；也可以像造物主那样，什么东西都用大脑来搞定: NLU-generate, NLU-copy, XXX-sort, 所以我们这个分类任务中的分解动作，逻辑是比较确定的，我们的意识也能感知到我们在这么处理问题。
