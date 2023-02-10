# TransRL
transformer in RL for decision-making 




# Transformer in Reinforcement Learning for Decision-Making: A Survey
Weilin Yuan, Jiaxing Chen, Shaofei Chen*, Lina Lu*, Zhenzhen Hu, Peng Li, Dawei Feng, Furong Liu Jing Chen


# Overview
 - [Transrl Methods](##Transrl)
 - [Representative Transrl Models](##Representative)
 - [Application](##Application)
 - [Challenges And Open Problems](##Challenges)

## Transrl Methods
**Efficient Transformers in Reinforcement Learning using Actor-Learner Distillation arXiv: Learning**[2021]
E. Parisotto and R. Salakhutdinov, [[PDF]](https://arxiv.org/pdf/2104.01655.pdf)

**Deep Transformer Q-Networks for Partially Observable Reinforcement Learning** [2022]
K. Esslinger, R. Platt, and C. Amato,  [[PDF]](https://export.arxiv.org/pdf/2206.01078.pdf)[[Github]](https://github.com/kevslinger/DTQN))

**Stabilizing Deep Q-Learning with ConvNets and Vision Transformers under Data Augmentation Cornell University - arXiv**[2021]
 Nicklas Hansen, Hao Su,  Xiaolong Wang[[PDF]](https://export.arxiv.org/pdf/2107.00644.pdf)[[Github]](https://nicklashansen.github.io/SVEA)
 
**Transformer Based Reinforcement Learning For Games.. arXiv: Learning**[2019]
 Uddeshya Upadhyay, Nikunj Shah, Sucheta Ravikanti, Mayanka Medhe [[PDF]](https://arxiv.org/abs/1912.03918)
 
**Training Agents using Upside-Down Reinforcement Learning**[2023]
 Rupesh Srivastava, Pranav Shyam, Filipe Mutz, Wojciech Jaśkowski, &Jürgen Schmidhuber [[PDF]](http://export.arxiv.org/pdf/1912.02877v1.pdf)
 
**Language Models are Few-Shot Learners arXiv: Computation and Language**[2020]
Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Thomas Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Samuel McCandlish, Alec Radford, Ilya Sutskever, & Dario Amodei  [[PDF]](https://proceedings.neurips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf)

**Offline Reinforcement Learning as One Big Sequence Modeling Problem **[2021]
Michael Janner, Qiyang Li, & Sergey Levine [[PDF]](http://export.arxiv.org/pdf/2106.02039v3.pdf) [[Github]](https://trajectory-transformer.github.io/)

**Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems arXiv: Learning.**[2020]
Sergey Levine, Aviral Kumar, George Tucker, & Justin Fu[[PDF]](https://export.arxiv.org/pdf/2005.01643.pdf)

**Decision Transformer: Reinforcement Learning via Sequence Modeling Cornell University - arXiv**[2021]
Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Michael Laskin, Pieter Abbeel, Aravind Srinivas, & Igor Mordatch [[PDF]](https://export.arxiv.org/pdf/2106.01345.pdf)[[Github1]](https://github.com/karpathy/minGPT)[[Github2]](https://github.com/karpathy/minGPT/blob/master/play_char.ipynb)

**Bootstrapped Transformer for Offline Reinforcement Learning**[2022]
Kerong Wang, Hanye Zhao, Xufang Luo, Kan Ren, Weinan Zhang, & Dongsheng Li  [[PDF]](https://export.arxiv.org/pdf/2206.08569v2.pdf)[[Github]](https://seqml.github.io/bootorl)

**Switch Trajectory Transformer with Distributional Value Approximation for Multi-Task Reinforcement Learning**[2022]
Qinjie Lin, Han Liu, & Biswa Sengupta[[PDF]](https://export.arxiv.org/pdf/2203.07413.pdf)

**Human-level Atari 200x faster.**[2022]
Steven Kapturowski, V\'ictor Campos, Ray Jiang, Nemanja Raki\'cevi\'c, Hado van Hasselt, Charles Blundell, & Adri\`a Puigdom\`enech Badia [[PDF]](https://export.arxiv.org/pdf/2209.07550v1.pdf)

**On the Opportunities and Risks of Foundation Models..” Cornell University - arXiv**[2021]
Rishi Bommasani et al.  [[PDF]](https://export.arxiv.org/pdf/2108.07258v3.pdf)

**Pretrained Transformers as Universal Computation Engines**[2021]
Kevin Lu, Aditya Grover, Pieter Abbeel, & Igor Mordatch[[PDF]](https://export.arxiv.org/pdf/2103.05247.pdf)[[Github]](https://github.com/kzl/universal-computation)

**Online Decision Transformer**[2022]
Qinqing Zheng, Amy Zhang, & Aditya Grover  [[PDF]](https://export.arxiv.org/pdf/2202.05607v2.pdf)

**Can Wikipedia Help Offline Reinforcement Learning?**[2022]
Machel Reid, Yutaro Yamada, & Shixiang Shane Gu[[PDF]](https://export.arxiv.org/pdf/2201.12122v3.pdf)[[Github]](https://github.com/machelreid/can-wikipedia-help-offline-rl)

**MineDojo: Building Open-Ended Embodied Agents with Internet-Scale Knowledge**[2022]
Linxi Fan, Guanzhi Wang, Yunfan Jiang, Ajay Mandlekar, Yuncong Yang, Haoyi Zhu, Andrew Tang, De-An Huang, Yuke Zhu, & Anima Anandkumar[[PDF]](https://export.arxiv.org/pdf/2206.08853v2.pdf)[[Github]](https://github.com/MineDojo/MineDojo)

**A Generalist Agent**[2022]
Scott Reed, Konrad Żołna, Emilio Parisotto, Sergio Gómez Colmenarejo, Alexander Novikov, Gabriel Barth-Maron, Mai Giménez, Yury Sulsky, Jackie Kay, Mahyar Bordbar, & Nando De Freitas [[PDF]](https://export.arxiv.org/pdf/2205.06175v3.pdf)[[Github]](https://github.com/rlworkgroup/metaworld/commit/a0009ed9a208ff9864a5c1368c04c273bb20dd06)

** Offline Pre-trained Multi-Agent Decision Transformer: One Big Sequence Model Tackles All SMAC Tasks**[2022]
Linghui Meng, Muning Wen, Yaodong Yang, Chenyang Le, Xiyun Li, Weinan Zhang, Ying Wen, Haifeng Zhang, Jun Wang, & Bo Xu [[PDF]](http://export.arxiv.org/pdf/2112.02845v3.pdf)[[Github]](https://github.com/ReinholdM/Offline-Pre-trained-Multi-Agent-Decision-Transformer)

**Gradient Surgery for Multi-Task Learning Cornell University**[2020]
Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, & Chelsea Finn [[PDF]](https://papers.nips.cc/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf)[[Github]](https://github.com/tianheyu927/PCGrad)

**Pretraining in Deep Reinforcement Learning: A Survey**[2022]
Zhihui Xie, Zichuan Lin, Junyou Li, Shuai Li, & Deheng Ye (2022). 
 [[PDF]](https://export.arxiv.org/pdf/2211.03959v1.pdf)
 
**Hierarchical Reinforcement Learning: A Comprehensive Survey ACM Computing Surveys**[2021]
Shubham Pateria, Budhitama Subagdja, Ah-Hwee Tan, & Chai Quek  [[PDF]](https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=7054&context=sis_research)




## Representative Transrl Models

## Application

## Challenges And Open Problems



