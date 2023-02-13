# TransRL
transformer in RL for decision-making 




# Transformer in Reinforcement Learning for Decision-Making: A Survey 

(we just upload partial references, and the left will be completed after our paper is published.)

# Overview

* [Transrl Methods](#transrl-methods)

     [1.Transformer-based Offline RL](#transformer-based-offline-rl) 
     
     [2.Transformer-based Online Reinforcement Learning](#transformer-based-online-reinforcement-learning) 
     
     [3.Trasnformer-based Hierarchical Reinforcement Learning](#trasnformer-based-hierarchical-reinforcement-learning)	 
     
     [4.Transformer-based Multi-agent Reinforcement Learning](#transformer-based-multi-agent-reinforcement-learning) 
    
     [5.Transformer-based Meta Reinforcement Learning](#transformer-based-meta-reinforcement-learning) 
    
* [Representative Transrl Models](#representative-transrl-models)

    [1.Decision Transformer ](#decision-transformer)
    
    [2.Multi-Agent Transformer (MAT)](#multi-agent-transformer)
    
    [3.Gato](#gato)
    
* [Application](#application)

    [1.Gaming AI](#gaming-ai)
    
    [2.Robotics](#robotics)
    
    [3.Transportation](#transportation)
      
    [4.Computer Systems](#computer-systems)
    
*  [Challenges And Open Problems](#challenges-and-open-problems)

     [1.Stability and Structure Optimization](#stability-and-structure-optimization)
     
     [2.Expensive Memory and Computation](#expensive-memory-and-computation)
     
     [3.Stochastic Effectiveness](#stochastic-effectiveness)
     
# Transrl Methods
## Transformer-based Offline RL

**Efficient Transformers in Reinforcement Learning using Actor-Learner Distillation**[2021]
E. Parisotto and R. Salakhutdinov[[PDF]](https://arxiv.org/pdf/2104.01655.pdf)

**Deep Transformer Q-Networks for Partially Observable Reinforcement Learning** [2022]
K. Esslinger, R. Platt, and C. Amato[[PDF]](https://export.arxiv.org/pdf/2206.01078.pdf)[[Github]](https://github.com/kevslinger/DTQN))

**Stabilizing Deep Q-Learning with ConvNets and Vision Transformers under Data Augmentation**[2021]
 Nicklas Hansen, Hao Su,  Xiaolong Wang[[PDF]](https://export.arxiv.org/pdf/2107.00644.pdf)[[Github]](https://nicklashansen.github.io/SVEA)
 
**Transformer Based Reinforcement Learning For Games**[2019]
 Uddeshya Upadhyay, Nikunj Shah, Sucheta Ravikanti, Mayanka Medhe [[PDF]](https://arxiv.org/abs/1912.03918)

**Training Agents using Upside-Down Reinforcement Learning**[2019]
 Rupesh Srivastava, Pranav Shyam, Filipe Mutz, Wojciech Jaśkowski, &Jürgen Schmidhuber [[PDF]](http://export.arxiv.org/pdf/1912.02877v1.pdf)

**Language Models are Few-Shot Learners**[2020]
Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Thomas Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Samuel McCandlish, Alec Radford, Ilya Sutskever, & Dario Amodei  [[PDF]](https://proceedings.neurips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf)

**Offline Reinforcement Learning as One Big Sequence Modeling Problem**[2021]
Michael Janner, Qiyang Li, & Sergey Levine [[PDF]](http://export.arxiv.org/pdf/2106.02039v3.pdf) [[Github]](https://trajectory-transformer.github.io/)

**Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems**[2020]
Sergey Levine, Aviral Kumar, George Tucker, & Justin Fu[[PDF]](https://export.arxiv.org/pdf/2005.01643.pdf)

**Decision Transformer: Reinforcement Learning via Sequence Modeling**[2021]
Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Michael Laskin, Pieter Abbeel, Aravind Srinivas, & Igor Mordatch [[PDF]](https://export.arxiv.org/pdf/2106.01345.pdf)[[Github1]](https://github.com/karpathy/minGPT)[[Github2]](https://github.com/karpathy/minGPT/blob/master/play_char.ipynb)

**Bootstrapped Transformer for Offline Reinforcement Learning**[2022]
Kerong Wang, Hanye Zhao, Xufang Luo, Kan Ren, Weinan Zhang, & Dongsheng Li  [[PDF]](https://export.arxiv.org/pdf/2206.08569v2.pdf)[[Github]](https://seqml.github.io/bootorl)

**Switch Trajectory Transformer with Distributional Value Approximation for Multi-Task Reinforcement Learning**[2022]
Qinjie Lin, Han Liu, & Biswa Sengupta[[PDF]](https://export.arxiv.org/pdf/2203.07413.pdf)

## Transformer-based Online Reinforcement Learning

**Human-level Atari 200x faster**[2022]
Steven Kapturowski, V\'ictor Campos, Ray Jiang, Nemanja Raki\'cevi\'c, Hado van Hasselt, Charles Blundell, & Adri\`a Puigdom\`enech Badia [[PDF]](https://export.arxiv.org/pdf/2209.07550v1.pdf)

**On the Opportunities and Risks of Foundation Models**[2021]
Rishi Bommasani et al.  [[PDF]](https://export.arxiv.org/pdf/2108.07258v3.pdf)

**Pretrained Transformers as Universal Computation Engines**[2021]
Kevin Lu, Aditya Grover, Pieter Abbeel, & Igor Mordatch[[PDF]](https://export.arxiv.org/pdf/2103.05247.pdf)[[Github]](https://github.com/kzl/universal-computation)

**Online Decision Transformer**[2022]
Qinqing Zheng, Amy Zhang, & Aditya Grover[[PDF]](https://export.arxiv.org/pdf/2202.05607v2.pdf)

**Can Wikipedia Help Offline Reinforcement Learning?**[2022]
Machel Reid, Yutaro Yamada, & Shixiang Shane Gu[[PDF]](https://export.arxiv.org/pdf/2201.12122v3.pdf)[[Github]](https://github.com/machelreid/can-wikipedia-help-offline-rl)

**MineDojo: Building Open-Ended Embodied Agents with Internet-Scale Knowledge**[2022]
Linxi Fan, Guanzhi Wang, Yunfan Jiang, Ajay Mandlekar, Yuncong Yang, Haoyi Zhu, Andrew Tang, De-An Huang, Yuke Zhu, & Anima Anandkumar[[PDF]](https://export.arxiv.org/pdf/2206.08853v2.pdf)[[Github]](https://github.com/MineDojo/MineDojo)

**A Generalist Agent**[2022]
Scott Reed, Konrad Żołna, Emilio Parisotto, Sergio Gómez Colmenarejo, Alexander Novikov, Gabriel Barth-Maron, Mai Giménez, Yury Sulsky, Jackie Kay, Mahyar Bordbar, & Nando De Freitas [[PDF]](https://export.arxiv.org/pdf/2205.06175v3.pdf)[[Github]](https://github.com/rlworkgroup/metaworld/commit/a0009ed9a208ff9864a5c1368c04c273bb20dd06)

**Offline Pre-trained Multi-Agent Decision Transformer: One Big Sequence Model Tackles All SMAC Tasks**[2022]
Linghui Meng, Muning Wen, Yaodong Yang, Chenyang Le, Xiyun Li, Weinan Zhang, Ying Wen, Haifeng Zhang, Jun Wang, & Bo Xu [[PDF]](http://export.arxiv.org/pdf/2112.02845v3.pdf)[[Github]](https://github.com/ReinholdM/Offline-Pre-trained-Multi-Agent-Decision-Transformer)

**Gradient Surgery for Multi-Task Learning**[2020]
Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, & Chelsea Finn [[PDF]](https://papers.nips.cc/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf)[[Github]](https://github.com/tianheyu927/PCGrad)

**Pretraining in Deep Reinforcement Learning: A Survey**[2022]
Zhihui Xie, Zichuan Lin, Junyou Li, Shuai Li, & Deheng Ye[[PDF]](https://export.arxiv.org/pdf/2211.03959v1.pdf)

## Trasnformer-based Hierarchical Reinforcement Learning

**Hierarchical Reinforcement Learning: A Comprehensive Survey ACM Computing Surveys**[2021]
Shubham Pateria, Budhitama Subagdja, Ah-Hwee Tan, & Chai Quek[[PDF]](https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=7054&context=sis_research)

**Hierarchical Decision Transformer**[2022]
Andr\'e Correia, & Lu\'is A. Alexandre[[PDF]](https://export.arxiv.org/pdf/2209.10447v1.pdf)

**Reinforcement Learning with Hierarchies of Machines**[1997]
Ronald Parr, & Stuart Russell[[PDF]](https://axon.cs.byu.edu/Dan/778/papers/Hierarchical%20Reinforcement%20Learning/Parr*.pdf)

**Learning Multi-Level Hierarchies with Hindsight**[2017]
Andrew Levy, George Konidaris, Robert W. Platt, & Kate Saenko [[PDF]](https://export.arxiv.org/pdf/1712.00948.pdf)[[Github]](https://github.com/andrew-j-levy/Hierarchical-Actor-Critc-HAC-)

## Transformer-based Multi-agent Reinforcement Learning

**An Overview of Multi-Agent Reinforcement Learning from Game Theoretical Perspective**[2020]
Yaodong Yang, & Jun Wang[[PDF]](https://arxiv.org/pdf/2011.00583.pdf)

**Multi-agent deep reinforcement learning: a survey**[2021]
Sven Gronauer, & Klaus Diepold[[PDF]](https://link.springer.com/content/pdf/10.1007/s10462-021-09996-w.pdf/n)

**The StarCraft Multi-Agent Challenge**[2019]
Mikayel Samvelyan, Tabish Rashid, Christian Schroeder de Witt, Gregory Farquhar, Nantas Nardelli, Tim G. J. Rudner, Chia-Man Hung, Philip H. S. Torr, Jakob Foerster, & Shimon Whiteson[[PDF]](https://export.arxiv.org/pdf/1902.04043)[[Github]](https://github.com/oxwhirl/smac)

**Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments**[2017]
Ryan Lowe, Yi Wu, Aviv Tamar, Jean Harb, OpenAI Pieter Abbeel, & Igor Mordatch[[PDF]](https://proceedings.neurips.cc/paper/2017/file/68a9750337a418a86fe06c1991a1d64c-Paper.pdf)[[Github]](https://sites.google.com/site/multiagentac/)

**Optimal and approximate Q-value functions for decentralized POMDPs**[2008]
Frans A. Oliehoek, Matthijs T. J. Spaan, & Nikos Vlassis[[PDF]](https://export.arxiv.org/pdf/1111.0062.pdf)

**Counterfactual Multi-Agent Policy Gradients**[2017]
Jakob Foerster, Gregory Farquhar, Triantafyllos Afouras, Nantas Nardelli, & Shimon Whiteson[[PDF]](https://arxiv.org/pdf/1705.08926.pdf)

**QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning**[2018]
Tabish Rashid, Mikayel Samvelyan, Christian Schroeder de Witt, Gregory Farquhar, Jakob Foerster, & Shimon Whiteson[[PDF]](https://export.arxiv.org/pdf/1803.11485.pdf)

**The Surprising Effectiveness of MAPPO in Cooperative, Multi-Agent Games**[2021]
Chao Yu, Akash Velu, Eugene Vinitsky, Yu Wang, Alexandre M. Bayen, & Yi Wu[[PDF]](https://export.arxiv.org/pdf/2103.01955v4.pdf)
[[Github]](https://github.com/marlbenchmark/on-policy)

**Value-Decomposition Networks For Cooperative Multi-Agent Learning Based On Team Reward**[2018]
Peter Sunehag, Guy Lever, Audrunas Gruslys, Wojciech Marian Czarnecki, Vinicius Zambaldi, Max Jaderberg, Marc Lanctot, Nicolas Sonnerat, Joel Z. Leibo, Karl Tuyls, & Thore Graepel[[PDF]](https://export.arxiv.org/pdf/1706.05296.pdf)

**Multi-Agent Determinantal Q-Learning**[2020]
Yaodong Yang, Ying Wen, Liheng Chen, Jun Wang, Kun Shao, David Mguni, & Weinan Zhang[[PDF]](https://arxiv.org/pdf/2006.01482.pdf)[[Github]](https://github.com/QDPP-GitHub/QDPP)

**Transformer-based Value Function Decomposition for Cooperative Multi-agent Reinforcement Learning in StarCraft**[2022]
Muhammad Junaid Khan, Syed Hammad Ahmed, & Gita Sukthanka[[PDF]](https://export.arxiv.org/pdf/2208.07298.pdf)[[Github]](https://github.com/QDPP-GitHub/QDPP)

**Transform networks for cooperative multi-agent deep reinforcement learning**[2022]
Hongbin Wang, • Xiaodong Xie, & Lianke Zhou [[PDF]](https://link.springer.com/article/10.1007/s10489-022-03924-3)

**UPDeT: Universal Multi-agent Reinforcement Learning via Policy Decoupling with Transformers**[2021]
Siyi Hu, Fengda Zhu, Xiaojun Chang, & Xiaodan Liang[[PDF]](http://export.arxiv.org/pdf/2101.08001v3.pdf)[[Github]](https://github.com/hhhusiyi-monash/UPDeT)

**Multi-Agent Reinforcement Learning is a Sequence Modeling Problem**[2022]
Muning Wen, Jakub Grudzien Kuba, Runji Lin, Weinan Zhang, Ying Wen, Jun Wang, & Yaodong Yang[[PDF]](https://export.arxiv.org/pdf/2205.14953v3.pdf)[[Github]](https://sites.google.com/view/multi-agent-transformer)

**Transformer-based Working Memory for Multiagent Reinforcement Learning with Action Parsing**[2022]
Yaodong Yang, Guangyong Chen, Weixun Wang, Xiaotian Hao, Jianye Hao, & Ann Heng [[PDF]](https://openreview.net/pdf?id=pd6ipu3jDw)[[Github]](https://github.com/CNDOTA/NeurIPS22-ATM)

**Meta-World: A Benchmark and Evaluation for Multi-Task and Meta Reinforcement Learning**[2019]
Tianhe Yu, Deirdre Quillen, Zhanpeng He, Ryan Julian, Avnish Narayan, Hayden Shively, Adithya Bellathur, Karol Hausman, Chelsea Finn, & Sergey Levine 
[[PDF]](https://export.arxiv.org/pdf/1910.10897.pdf)[[Github]](https://github.com/rlworkgroup/metaworld)

**Meta-Learning in Neural Networks: A Survey**[2020]
Timothy M. Hospedales, Antreas Antoniou, Paul Micaelli, & Amos Storkey[[PDF]](https://arxiv.org/pdf/2004.05439/n)

## Transformer-based Meta Reinforcement Learning

**Meta-learning of Sequential Strategies**[2019]
Pedro A. Ortega, Jane X. Wang, Mark Rowland, Tim Genewein, Zeb Kurth-Nelson, Razvan Pascanu, Nicolas Heess, Joel Veness, Alexander Pritzel, Pablo Sprechmann, Siddhant M. Jayakumar, Thomas M McGrath, Kevin J. Miller, Mohammad Gheshlaghi Azar, Ian Osband, Neil C. Rabinowitz, András György, Silvia Chiappa, Simon Osindero, Yee Whye Teh, Hado van Hasselt, Nando de Freitas, Matthew Botvinick, & Shane Legg[[PDF]](http://export.arxiv.org/pdf/1905.03030v2.pdf)

**RL^2: Fast Reinforcement Learning via Slow Reinforcement Learning**[2016]
Yan Duan, John Schulman, Xi Chen, Peter L. Bartlett, Ilya Sutskever, & Pieter Abbeel [[PDF]](https://arxiv.org/pdf/1611.02779.pdf)

**Some Considerations on Learning to Explore via Meta-Reinforcement Learning**[2018]
Bradly C. Stadie, Ge Yang, Rein Houthooft, Xi Chen, Yan Duan, Yuhuai Wu, Pieter Abbeel, & Ilya Sutskever[[PDF]](https://openreview.net/pdf?id=Skk3Jm96W)

**Transformers are Meta-Reinforcement Learners**[2022]
Luckeciano C. Melo
[[PDF]](https://export.arxiv.org/pdf/2206.06614.pdf)[[Github]](https://github.com/luckeciano/transformers-metarl)

**A model-based approach to meta-Reinforcement Learning: Transformers and tree search**[2022]
Brieuc Pinon, Jean-Charles Delvenne, & Rapha\"el Jungers[[PDF]](https://export.arxiv.org/pdf/2208.11535v1.pdf)

**Alchemy: A benchmark and analysis toolkit for meta-reinforcement learning agents**[2021]
Jane X. Wang, Michael A. King, Nicolas Porcel, Zeb Kurth-Nelson, Tina Zhu, Charlie Deck, Peter Choy, Mary Cassin, Malcolm Reynolds, H. Francis Song, Gavin Buttimore, David P. Reichert, Neil C. Rabinowitz, Loic Matthey, Demis Hassabis, Alexander Lerchner, & Matthew Botvinick [[PDF]](https://openreview.net/pdf?id=pd6ipu3jDw)[[Github]](https://github.com/CNDOTA/NeurIPS22-ATM)

**Contextual Transformer for Offline Meta Reinforcement Learning**[2022]
Runji Lin, Ye Li, Xidong Feng, Zhaowei Zhang, Xian Hong Wu Fung, Haifeng Zhang, Jun Wang, Yali Du, & Yaodong Yang[[PDF]](https://export.arxiv.org/pdf/2211.08016v1.pdf)

**Prompting Decision Transformer for Few-Shot Policy Generalization**[2022]
Mengdi Xu, Yikang Shen, Shun Zhang, Yuchen Lu, Ding Zhao, Joshua B Tenenbaum, & Chuang Gan[[PDF]](https://export.arxiv.org/pdf/2206.13499.pdf)[[Github]](https://mxu34.github.io/PromptDT/)

**Offline Meta-Reinforcement Learning with Advantage Weighting**[2020]
Eric Mitchell, Rafael Rafailov, Xue Bin Peng, Sergey Levine, & Chelsea Finn[[PDF]](https://openreview.net/pdf?id=S5S3eTEmouw)[[Github]](https://github.com/rlworkgroup/metaworld/issues/226)

# Representative Transrl Models

### Decision Transformer
**An Image is Worth 16x16 Words: Transformers for Image Recognition at Scal**[2020]
Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, & Neil Houlsby[[PDF]](https://openreview.net/pdf?id=YicbFdNTTy)[[Github]](https://github.com/google-research/vision_transformer)

**How Crucial is Transformer in Decision Transformer?**[2022]
Max Siebenborn, Boris Belousov, Junning Huang, & Jan Peters[[PDF]](https://export.arxiv.org/pdf/2211.14655v1.pdf)

**Q-learning Decision Transformer: Leveraging Dynamic Programming for Conditional Sequence Modelling in Offline RL**[2022]
Taku Yamagata, Ahmed Khalil, Raul Santos-Rodriguez (Intelligent System Laboratory, & University of Bristol)[[PDF]](https://export.arxiv.org/pdf/2209.03993v3.pdf)[[Github]](https://github.com/kzl/decision-transformer)

**StARformer: Transformer with State-Action-Reward Representations for Visual Reinforcement Learning**[2021]
Jinghuan Shang, Kumara Kahatapitiya, Xiang Li, & Michael S. Ryoo[[PDF]](https://export.arxiv.org/pdf/2110.06206v3.pdf)[[Github]](https://github.com/elicassion/StARformer)

**Generalized Decision Transformer for Offline Hindsight Information Matching**[2022]
Hiroki Furuta, Yutaka Matsuo, & Shixiang Shane Gu[[PDF]](http://export.arxiv.org/pdf/2111.10364v3.pdf)[[Github]](https://github.com/frt03/generalized_dt)

## Multi-Agent Transformer

**Settling the Variance of Multi-Agent Policy Gradients**[2021]
Jakub Grudzien Kuba, Muning Wen, Linghui Meng, Shangding Gu, Haifeng Zhang, David Mguni, Jun Wang, & Yaodong Yang[[PDF]](https://openreview.net/pdf?id=pd6ipu3jDw)

**Reinforcement Learning: An Introduction**[1988]
Richard S. Sutton, & Andrew G. Barto (1988)[[PDF]](https://readpaper.com/pdf-annotate/note?pdfId=4509452219912445953&noteId=1649148406125379072)

**Proximal Policy Optimization Algorithms**[2017]
John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, & Oleg Klimov[[PDF]](https://arxiv.org/pdf/1707.06347.pdf)[[Github]](https://github.com/berkeleydeeprlcourse/homework/tree/master/hw4)

## Gato
**The arcade learning environment: an evaluation platform for general agents**[2015]
Marc G. Bellemare, Yavar Naddaf, Joel Veness, & Michael Bowling[[PDF]](https://export.arxiv.org/pdf/1207.4708.pdf)

**SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing Empirical Methods in Natural Language Processing**[2018]
Taku Kudo, & John Richardson[[PDF]](https://arxiv.org/pdf/1808.06226.pdf)[[Github]](https://github.com/google/sentencepiece)
# Application

**Recent Advances in Deep Reinforcement Learning Applications for Solving Partially Observable Markov Decision Processes**[2021]
Xuanchen Xiang, & Simon Foo (2021)[[PDF]](https://readpaper.com/pdf-annotate/note?pdfId=4546315169662918657&noteId=1648136071059422720)

## Gaming AI
**Imperfect Information Game in Multiplayer No-limit Texas Hold’em Based on Mean Approximation and Deep CFVnet**[2021]
Yuan Weilin, Hu Zhenzhen, Luo Junren, Xu Jiahui, Ji Xiang, Chen Shaofei, Zhang Wanpeng, & Chen Jing[[PDF]](https://readpaper.com/pdf-annotate/note?pdfId=4546315169662918657&noteId=1648136071059422720)

**Navigating the Landscape of Games**[2020]
Shayegan Omidshafiei, Karl Tuyls, Wojciech Marian Czarnecki, Francisco C. Santos, Mark Rowland, Jerome T. Connor, Daniel Hennes, Paul Muller, Julien Perolat, Bart De Vylder, Audrunas Gruslys, & Rémi Munos[[PDF]](https://arxiv.org/pdf/2005.01642.pdf)[[Github]](https://github.com/deepmind/open_spie)

**Agent57: Outperforming the Atari Human Benchmark**[2020]
Adrià Puigdomènech Badia, Bilal Piot, Steven Kapturowski, Pablo Sprechmann, Alex Vitvitskyi, Daniel Guo, & Charles Blundell[[PDF]](https://arxiv.org/pdf/2005.01642.pdf)[[Github]](https://github.com/mgbellemare/Arcade-Learning-Environment)

**Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm**[2017]
David Silver, Thomas Hubert, Julian Schrittwieser, Ioannis Antonoglou, Matthew Lai, Arthur Guez, Marc Lanctot, Laurent Sifre, Dharshan Kumaran, Thore Graepel, Timothy P. Lillicrap, Karen Simonyan, & Demis Hassabis[[PDF]](https://arxiv.org/pdf/1712.01815.pdf)

**Human-level performance in 3D multiplayer games with population-based reinforcement learning**[2019]
Max Jaderberg, Wojciech Marian Czarnecki, Iain Dunning, Luke Marris, Guy Lever, Antonio García Castañeda, Charles Beattie, Neil C. Rabinowitz, Ari S. Morcos, Avraham Ruderman, Nicolas Sonnerat, Tim Green, Louise Deason, Joel Z. Leibo, David Silver, Demis Hassabis, Koray Kavukcuoglu, & Thore Graepel[[PDF]](https://pubmed.ncbi.nlm.nih.gov/31147514/)

**Suphx: Mastering Mahjong with Deep Reinforcement Learning**[2020]
Junjie Li, Sotetsu Koyamada, Qiwei Ye, Guoqing Liu, Chao Wang, Ruihan Yang, Li Zhao, Tao Qin, Tie-Yan Liu, & Hsiao-Wuen Hon[[PDF]](https://arxiv.org/pdf/2003.13590/n)

**AlphaHoldem: High-Performance Artificial Intelligence for Heads-Up No-Limit Texas Hold'em from End-to-End Reinforcement Learning**[2022]
Enmin Zhao, Renye Yan, Jinqiu Li, Kai Li, & Junliang Xing [[PDF]](https://ojs.aaai.org/index.php/AAAI/article/view/20394)

**DouZero: Mastering DouDizhu with Self-Play Deep Reinforcement Learning**[2021]
Daochen Zha, Jingru Xie, Wenye Ma, Sheng Zhang, Xiangru Lian, Xia Hu, & Ji Liu[[PDF]](https://export.arxiv.org/pdf/2106.06135.pdf)[[Github]](https://github.com/kwai/DouZero)

**Grandmaster level in starcraft ii using multi-agent reinforcement learning**[2019]
Oriol Vinyals, Igor Babuschkin, Wojciech Czarnecki, Michaël Mathieu, Andrew Dudzik, Junyoung Chung, David Choi, Richard Powell, Timo Ewalds, Petko Georgiev, Junhyuk Oh, Dan Horgan, Manuel Kroiss, Ivo Danihelka, Aja Huang, Laurent Sifre, Trevor Cai, John Agapiou, Max Jaderberg, Alexander Vezhnevets, Rémi Leblond, Tobias Pohlen, Valentin Dalibard, David Budden, Yury Sulsky, James Molloy, Tom Paine, Caglar Gulcehre, Ziyu Wang, Tobias Pfaff, Yuhuai Wu, Roman Ring, Dani Yogatama, Dario Wünsch, Katrina Mckinney, Oliver Smith, Tom Schaul, Timothy Lillicrap, Koray Kavukcuoglu, Demis Hassabis, Chris Apps, & David Silver[[PDF]](https://www.nature.com/articles/s41586-019-1724-z)

**OpenAI Gym**[2016]
Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, & Wojciech Zaremba[[PDF]](https://export.arxiv.org/pdf/1606.01540.pdf)

**Google Research Football: A Novel Reinforcement Learning Environment**[2019]
Karol Kurach, Anton Raichuk, Piotr Stańczyk, Michał Zając, Olivier Bachem, Lasse Espeholt, Carlos Riquelme, Damien Vincent, Marcin Michalski, Olivier Bousquet, & Sylvain Gelly[[PDF]](https://export.arxiv.org/pdf/1907.11180.pdf)

**Real World Games Look Like Spinning Tops**[2020]
Wojciech Marian Czarnecki, Gauthier Gidel, Brendan D. Tracey, Karl Tuyls, Shayegan Omidshafiei, David Balduzzi, & Max Jaderberg[[PDF]](https://export.arxiv.org/pdf/2004.09468)

**MineRL: A Large-Scale Dataset of Minecraft Demonstrations**[2019]
William H. Guss, Brandon Houghton, Nicholay Topin, Phillip Wang, Cayden Codel, Manuela Veloso, & Ruslan Salakhutdinov[[PDF]](https://export.arxiv.org/pdf/1907.13440.pdf)
**Open-Ended Evolution for Minecraft Building Generation**[2022]
Matthew Barthet, Antonios Liapis, & Georgios N. Yannakakis[[PDF]](https://export.arxiv.org/pdf/2209.03108v1.pdf)
## Robotics

**Reinforcement learning in robotic applications: a comprehensive survey**[2021]Bharat Singh, Rajesh Kumar, & Vinay Pratap Singh[[PDF]](https://link.springer.com/article/10.1007/s10462-021-09997-9)

**TAX-Pose: Task-Specific Cross-Pose Estimation for Robot Manipulation**[2022]
Chuer Pan, Brian Okorn, Harry Zhang, Ben Eisner, & David Held[[PDF]](https://export.arxiv.org/pdf/2211.09325v1.pdf)

**Toward mobile robots reasoning like humans**[2015]
Jean Oh, Arne Suppe, Felix Duvallet, Abdeslam Boularias, Luis E. Navarro-Serment, Martial Hebert, Anthony Stentz, Jerry Vinokurov, Oscar J. Romero, Christian Lebiere, & Robert Dean[[PDF]](https://www.ri.cmu.edu/pub_files/2015/1/rcta-aaai2015.pdf)

**Grounding spatial relations for outdoor robot navigation**[2015]
Abdeslam Boularias, Felix Duvallet, Jean Oh, & Anthony Stentz[[PDF]](https://www.ri.cmu.edu/pub_files/2015/5/groundingICRA.pdf)

**Inferring Maps and Behaviors from Natural Language Instructions**[2015]
Felix Duvallet, Jean Oh, Anthony Stentz, Matthew R. Walter, Thomas M. Howard, Sachithra Hemachandra, Seth Teller, & Nicholas Roy[[PDF]](https://people.csail.mit.edu/mwalter/papers/duvallet14.pdf)

**Leveraging Transformers to Collect Robotic Task Demonstrations**[2021]
Henry M. Clever, Ankur Handa, Hammad Mazhar, Kevin Parker, Omer Shapira, Qian Wan, Yashraj Narang, Iretiayo Akinola, Maya Cakmak, & Dieter Fox [[PDF]](https://export.arxiv.org/pdf/2112.05129.pdf)[[Github]](https://sites.google.com/view/assistive-teleop)

**Assistive Tele-op: Leveraging Transformers to Collect Robotic Task Demonstrations**[2021]
Henry M. Clever, Ankur Handa, Hammad Mazhar, Kevin Parker, Omer Shapira, Qian Wan, Yashraj Narang, Iretiayo Akinola, Maya Cakmak, & Dieter Fox[[PDF]](https://export.arxiv.org/pdf/2112.05129.pdf)

**Learning Vision-Guided Quadrupedal Locomotion End-to-End with Cross-Modal Transformers**[2021]
Ruihan Yang, Minghao Zhang, Nicklas Hansen, Huazhe Xu, & Xiaolong Wang[[PDF]](http://export.arxiv.org/pdf/2107.03996v3.pdf)[[Github]](https://rchalyang.github.io/LocoTransformer/.)

**Learning Generalizable Vision-Tactile Robotic Grasping Strategy for Deformable Objects via Transformer**[2023]
Yunhai Han, Rahul Batra, Nathan Boyd, Tuo Zhao, Yu She, Seth Hutchinson, & Ye Zhao[[PDF]](https://export.arxiv.org/pdf/2112.06374v4.pdf)[[Github]](https://github.com/GTLIDAR/DeformableObjectsGrasping.git)

**Towards advanced robotic manipulation**[2022]
Francisco Roldan Sanchez, Stephen Redmond, Kevin McGuinness, & Noel O'Connor[[PDF]](https://export.arxiv.org/pdf/2209.08903v2.pdf)

**Counterfactual Credit Assignment in Model-Free Reinforcement Learning**[2021]
Thomas Mesnard, Theophane Weber, Fabio Viola, Shantanu Thakoor, Alaa Saade, Anna Harutyunyan, Will Dabney, Thomas Stepleton, Nicolas Heess, Arthur Guez, Eric Moulines, Marcus Hutter, Lars Buesing, & Rémi Munos[[PDF]](http://export.arxiv.org/pdf/2011.09464v2.pdf)

**Deep Multi-Agent Reinforcement Learning for Decentralized Continuous Cooperative Control**[2020]
Christian Schroeder de Witt, Bei Peng, Pierre-Alexandre Kamienny, Philip H. S. Torr, Wendelin Böhmer, & Shimon Whiteson[[PDF]](https://export.arxiv.org/pdf/2003.06709/n)[[Github]](https://github.com/schroederdewitt/multiagent-particle-envs/)

**Efficient Spatiotemporal Transformer for Robotic Reinforcement Learning**[2022]
Yiming Yang, Dengpeng Xing, & Bo Xu[[PDF]](https://www.researchgate.net/publication/361582649_Efficient_Spatiotemporal_Transformer_for_Robotic_Reinforcement_Learning)

**CALVIN: A Benchmark for Language-conditioned Policy Learning for Long-horizon Robot Manipulation Tasks**[2022]
Oier Mees, Lukas Hermann, Erick Rosete-Beas, & Wolfram Burgard[[PDF]](https://export.arxiv.org/pdf/2112.03227v4.pdf)

**Grounded Language Learning in a Simulated 3D World**[2017]
Karl Moritz Hermann, Felix Hill, Simon Green, Fumin Wang, Ryan Faulkner, Hubert Soyer, David Szepesvari, Wojciech Marian Czarnecki, Max Jaderberg, Denis Teplyashin, Marcus Wainwright, Chris Apps, Demis Hassabis, & Phil Blunsom[[PDF]](https://arxiv.org/pdf/1706.06551.pdf)

**Transformers are Adaptable Task Planners**[2022]
Vidhi Jain, Yixin Lin, Eric Undersander, Yonatan Bisk, & Akshara Rai[[PDF]](https://export.arxiv.org/pdf/2207.02442.pdf)

**Instruction-driven history-aware policies for robotic manipulations**[2022]
P.-L. Guhur, S. Chen, R. Garcia, M. Tapaswi, I. Laptev, and C. Schmid[[PDF]](https://export.arxiv.org/pdf/2209.04899v2.pdf)[[Github]](https://guhur.github.io/hiveformer/)

**Learning to Navigate in Interactive Environments with the Transformer-based Memory**[2022]
W. Li, R. Hong, J. Shen, and Y. Lu[[PDF]](https://embodied-ai.org/papers/2022/24.pdf)[[PDF]](https://embodied-ai.org/papers/2022/24.pdf)

**Vision-and-language navigation: Interpreting visually-grounded navigation instructions in real environments**[2017]
 P. Anderson, Q. Wu, D. Teney, J. Bruce, M. Johnson, N. S ̈underhauf, I. Reid, S. Gould, and A. van den Hengel[[PDF]](https://arxiv.org/pdf/1711.07280.pdf)


**Vision-and-Language Navigation: A Survey of Tasks, Methods, and Future Directions**[2022]
J. Gu, E. Stefani, Q. Wu, J. Thomason, and X. E. Wang[[PDF]](http://export.arxiv.org/pdf/2203.12667v3.pdf)[[Github]](https://github.com/eric-ai-lab/awesome-vision-language-navigation)

**Vision-and-language navigation: Interpreting visually-grounded navigation instructions in real environments**[2018]
P. Anderson, Q. Wu, D. Teney, J. Bruce, M. Johnson, N. S ̈underhauf, I. Reid, S. Gould, and A. van den Hengel[[PDF]](https://arxiv.org/pdf/1711.07280.pdf)

**Incorporating External Knowledge Reasoning for Vision-and-Language Navigation with Assistant's Help**[2022]
V. A. Cicirello, X. Li, Y. Zhang, W. Yuan, and J. Luo[[PDF]](https://www.researchgate.net/publication/361992594_Incorporating_External_Knowledge_Reasoning_for_Vision-and-Language_Navigation_with_Assistant%27s_Help)

**Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: a Survey**[2020]
W. Zhao, J. P. Queralta, and T. Westerlund[[PDF]](https://export.arxiv.org/pdf/2009.13303.pdf)

**ManipulaTHOR: A Framework for Visual Object Manipulation**[2021]
K. Ehsani, W. Han, A. Herrasti, E. VanderBilt, L. Weihs, E. Kolve, A. Kembhavi, and R. Mottaghi[[PDF]](http://export.arxiv.org/pdf/2104.11213v1.pdf)

**dm_control: Software and Tasks for Continuous Control**[2020]
Y. Tassa, S. Tunyasuvunakool, A. Muldal, Y. Doron, P. Trochim, S. Liu, S. Bohez, J. Merel, T. Erez, T. P. Lillicrap, and N. Heess[[PDF]](https://export.arxiv.org/pdf/2006.12983.pdf)[[Github]](https://www.github.com/deepmind/dm_control)

**SAPIEN: A SimulAted Part-Based Interactive ENvironment**[2020]
F. Xiang, Y. Qin, K. Mo, Y. Xia, H. Zhu, F. Liu, M. Liu, H. Jiang, Y. Yuan, H. Wang, L. Yi, A. X. Chang, L. J. Guibas, and H. Su[[PDF]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Xiang_SAPIEN_A_SimulAted_Part-Based_Interactive_ENvironment_CVPR_2020_paper.pdf)

**CausalWorld: A Robotic Manipulation Benchmark for Causal Structure and Transfer Learning**[2020]
O. Ahmed, F. Tr ̈auble, A. Goyal, A. Neitz, Y. Bengio, B. Sch ̈olkopf, M. W ̈uthrich, and S. Bau[[PDF]](https://export.arxiv.org/pdf/2010.04296/n)]

**RLBench: The Robot Learning Benchmark & Learning Environment**[2020]
S. James, Z. Ma, D. R. Arrojo, and A. J. Davison[[PDF]](https://arxiv.org/pdf/1909.12271.pdf)
## Transportation

**A Survey of Autonomous Driving: Common Practices and Emerging Technologies**[2020]
E. Yurtsever, J. Lambert, A. Carballo, and K. Takeda[[PDF]](https://arxiv.org/abs/1906.05113)

**Augmenting Reinforcement Learning with Transformer-based Scene Representation Learning for Decision-making of Autonomous Driving**[2022]
H. Liu, Z. Huang, X. Mo, and C. Lv[[PDF]](https://export.arxiv.org/pdf/2208.12263v1.pdf)

**Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor**[2018]
T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine[[PDF]](https://export.arxiv.org/pdf/1801.01290.pdf)[[Github]](http://github.com/haarnoja/sac)

**T 3 OMVP: A Transformer-Based Time and Team Reinforcement Learning Scheme for Observation-Constrained Multi-Vehicle Pursuit in Urban Area**[2021]
Z. Yuan, T. Wu, Q. Wang, Y. Yang, L. Li, and L. Zhang[[PDF]](https://arxiv.org/abs/2203.00183) 

**Applications of Integrated IoT-Fog-Cloud Systems to Smart Cities: A Survey**[2021]
N. Mohamed, J. Al-Jaroodi, S. Lazarova-Molnar, and I. Jawhar[[PDF]](https://www.researchgate.net/publication/356606381_Applications_of_Integrated_IoT-Fog-Cloud_Systems_to_Smart_Cities_A_Survey)

**A Traffic-Aware Federated Imitation Learning Framework for Motion Control at Unsignalized Intersections with Internet of Vehicles**[2021]
T. Wu, M. Jiang, Y. Han, Z. Yuan, X. Li, and L. Zhang[[PDF]](https://www.researchgate.net/publication/356846339_A_Traffic-Aware_Federated_Imitation_Learning_Framework_for_Motion_Control_at_Unsignalized_Intersections_with_Internet_of_Vehicles)[[Github]](https://github.com/shogun2015/TAFI-MC)

**Addressing Optimism Bias in Sequence Modeling for Reinforcement Learning**[2022]
A. Villaflor, Z. Huang, S. Pande, J. Dolan, and J. Schneidert[[PDF]](https://export.arxiv.org/pdf/2207.10295.pdf)[[Github]](https://github.com/avillaflor/SPLTtransforme)
## Computer Systems

**The Transformer Network for the Traveling Salesman Problem**[2021]
X. Bresson and T. Laurent[[PDF]](https://export.arxiv.org/pdf/2103.03012.pdf)[[Github]](https://github.com/xbresson/TSP_Transformer)

**Attention, Learn to Solve Routing Problems!**[2018]
W. Kool, H. van Hoof, and M. Welling[[PDF]](https://export.arxiv.org/pdf/1803.08475.pdf)[[Github]](https://github.com/wouterkool/attention-learn-to-route)

**Deep reinforcement learning for transportation network combinatorial optimization: A survey**[2021]
Q. Wang and C. Tang[[PDF]](https://www.sciencedirect.com/science/article/abs/pii/S0950705121007887?via%3Dihub)

**Learning Combinatorial Optimization on Graphs: A Survey With Applications to Networking**[2020]
N. Vesselinova, R. Steinert, D. F. Perez-Ramirez, and M. Boman[[PDF]](https://arxiv.org/pdf/2005.11081.pdf)

**Learning Heuristics for the TSP by Policy Gradient**[2018]
M. Deudon, P. Cournut, A. Lacoste, Y. Adulyasak, and L.-M. Rousseau[[PDF]](https://link.springer.com/chapter/10.1007/978-3-319-93031-2_12#Fn1)
[[Github]](https://github.com/MichelDeudon/encode-attend-navigate)

**Computers and Intractability: A Guide to the Theory of NP-Completeness**[1979]
M. R. Garey and D. S. Johnson[[PDF]](https://epubs.siam.org/doi/10.1137/1024022)

**Integer programming**[1972]
G. L. Nemhauser and L. A. Wolsey[[PDF]](https://onlinelibrary.wiley.com/doi/book/10.1002/9781119606475)

**Neural Combinatorial Optimization with Reinforcement Learning**[2016]
I. Bello, H. Pham, Q. V. Le, M. Norouzi, and S. Bengio[[PDF]](https://arxiv.org/pdf/1611.09940.pdf)

**Vehicle routing: Problems, methods, and applications, second edition**[2014]
P. Toth and D. Vigo[[PDF]](https://dl.acm.org/doi/10.5555/2723809)

**The Orienteering Problem**[1987]
B. L. Golden, L. Levy, and R. Vohra[[PDF]](https://www.semanticscholar.org/paper/The-Orienteering-Problem-Golden-Levy/08bd46a1d89554868d496ec7ebae782d2af5f817)

**Survey of Methodologies for TSP and VRP**[2014]
S. Anbuudayasankar, K. Ganesh, and S. Mohapatra[[PDF]](https://link.springer.com/chapter/10.1007/978-3-319-05035-5_2)

**Reinforcement learning for combinatorial optimization: A survey**[2021]
N. Mazyavkina, S. Sviridov, S. Ivanov, and E. Burnaev[[pdf]](https://arxiv.org/pdf/2003.03600.pdf)

**Sample Efficient Reinforcement Learning with REINFORCE**[2021]
J. Zhang, J. Kim, B. O'Donoghue, and S. Boyd, [[pdf]](https://export.arxiv.org/pdf/2010.11364.pdf)

**Learning Improvement Heuristics for Solving Routing Problems**[2019]
Y. Wu, W. Song, Z. Cao, J. Zhang, and A. Lim[[PDF]](https://arxiv.org/pdf/1912.05784v2.pdf)

# Challenges And Open Problems
## Stability and Structure Optimization

**Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned**[2019]
Elena Voita, David Talbot, Fedor Moiseev, Rico Sennrich, & Ivan Titov[[PDF]](https://arxiv.org/pdf/1905.09418v2.pdf)[[Github]](https://github.com/lena-voita/the-story-of-heads)

**On Layer Normalizations and Residual Connections in Transformers**[2022]
S. Takase, S. Kiyono, S. Kobayashi, and J. Suzuki[[PDF]](https://export.arxiv.org/pdf/2206.00330.pdf)[[Github]](https://github.com/pytorch/fairseq)

**Analyzing Attention Mechanisms through Lens of Sample Complexity and Loss Landscape**[2021]
B. Liu, Y. Balaji, L. Xue, and M. R. Min[[PDF]](https://openreview.net/pdf?id=8KhxoxKP3iL)

**Attention Interpretability Across NLP Tasks**[2019]
S. Vashishth, S. Upadhyay, G. S. Tomar, and M. Faruqui[[PDF]](https://arxiv.org/pdf/1909.11218.pdf)[[Github]](https://github.com/tensorflow/tensorflow/issues/6269)

**Attention is Not All You Need: Pure Attention Loses Rank Doubly Exponentially with Dept**[2021]
Yihe Dong, Jean-Baptiste Cordonnier, & Andreas Loukas[[PDF]](https://arxiv.org/pdf/2103.03404.pdf)[[Github]](https://github.com/twistedcubic/attention-rank-collapse)

**Convexifying Transformers: Improving optimization and understanding of transformer networks**[2022]
T. Ergen, B. Neyshabur, and H. Mehta[[PDF]](https://export.arxiv.org/pdf/2211.11052v1.pdf)

**Catformer: Designing Stable Transformers via Sensitivity Analysis**[2021]
J. Davis, A. Gu, K. Choromanski, T. Dao, C. R ́e, C. Finn, and P. Liang [[PDF]](http://proceedings.mlr.press/v139/davis21a/davis21a.pdf)

**Understanding the difficulty of training transformers**[2020]
L. Liu, X. Liu, J. Gao, W. Chen, and J. Han[[PDF]](https://arxiv.org/abs/2004.08249#:~:text=Understanding%20the%20Difficulty%20of%20Training%20Transformers.%20Transformers%20have,%28the%20standard%20SGD%20fails%20to%20train%20Transformers%20effectively%29.)[[Github]](https://github.com/LiyuanLucasLiu/Transforemr-Clinic)

**Adaptive Input Representations for Neural Language Modeling**[2018]
A. Baevski and M. Auli[[PDF]](https://openreview.net/pdf?id=ByxZX20qFQ)[[Github]](http://github.com/pytorch/fairseq)

**On Layer Normalization in the Transformer Architecture**[2020]
R. Xiong, Y. Yang, D. He, K. Zheng, S. Zheng, C. Xing, H. Zhang, Y. Lan, L. Wang, and T.-Y. Liu[[PDF]](https://arxiv.org/pdf/2002.04745.pdf)

**Catformer: Designing Stable Transformers via Sensitivity Analysis**[2021]
J. Davis, A. Gu, K. Choromanski, T. Dao, C. R ́e, C. Finn, and P.Liang[[PDF]](http://proceedings.mlr.press/v139/davis21a/davis21a.pdf)

**Understanding the Difficulty of Training Transformers**[2020]
L. Liu, X. Liu, J. Gao, W. Chen, and J.A. Baevski and M Han[[PDF]](https://arxiv.org/pdf/2004.08249.pdf)[[Github]](https://github.com/LiyuanLucasLiu/Transforemr-Clinic) 

**An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale**[2020]
A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby[[PDF]](https://readpaper.com/pdf-annotate/note?noteId=1648120269757300224)[[Github]](https://github.com/csrhddlam/axial-deeplab)

**The Arcade Learning Environment: An Evaluation Platform for General Agents**[2013]
M. G. Bellemare, Y. Naddaf, J. Veness, and M. Bowling[[PDF]](https://arxiv.org/pdf/1207.4708.pdf)

**D4RL: Datasets for Deep Data-Driven Reinforcement Learning**[2021]
J. Fu, A. Kumar, O. Nachum, G. Tucker, and S. Levine[[PDF]](https://arxiv.org/pdf/2004.07219.pdf)

**Speech understanding systems: summary of results of the five-year research effort at Carnegie-Mellon University**[1977]
C. M. U. C. S. Dept[[PDF]](https://kilthub.cmu.edu/articles/journal_contribution/Speech_understanding_systems_summary_of_results_of_the_five-year_research_effort_at_Carnegie-Mellon_University_/6609821/1)

**Learning policies for partially observable environments: scaling up**[1997]
 M. L. Littman, A. R. Cassandra, and L. P. Kaelbling[[PDF]](https://www.sciencedirect.com/science/article/pii/B9781558603776500529)

**Imagination-Augmented Agents for Deep Reinforcement Learning**[2018]
S. Racani`ere, T. Weber, D. P. Reichert, L. Buesing, A. Guez, D. J. Rezende, A. P. Badia, O. Vinyals, N. Heess, Y. Li, R. Pascanu, P. W. Battaglia, D. Hassabis, D. Silver, and D. Wierstra[[PDF]](https://readpaper.com/pdf-annotate/note?pdfId=4500306879196061697&noteId=1651659371932892672)

**DeepMind Lab**[2016]
C. Beattie, J. Z. Leibo, D. Teplyashin, T. Ward, M. Wainwright, H. K ̈uttler, A. Lefrancq, S. Green, V. Vald ́es, A. Sadik, J. Schrittwieser, K. Anderson, S. York, M. Cant, A. Cain, A. Bolton, S. Gaffney, H. King, D. Hassabis, S. Legg, and S. Petersen[[PDF]](https://arxiv.org/pdf/1612.03801v2/n)

**BabyAI: A Platform to Study the Sample Efficiency of Grounded Language Learning**[2019]
M. Chevalier-Boisvert, D. Bahdanau, S. Lahlou, L. Willems, C. Saharia, T. H. Nguyen, and Y.Bengio[[PDF]](http://export.arxiv.org/pdf/1810.08272v3.pdf)[[Github]](https://github.com/mila-iqia/babyai/tree/iclr19)

**What Matters in Learning from Offline Human Demonstrations for Robot Manipulation**[2021]
A. Mandlekar, D. Xu, J. Wong, S. Nasiriany, C. Wang, R. Kulkarni, L. Fei-Fei, S. Savarese, Y. Zhu, and R. Mart ́ın-Mar[[PDF]](http://export.arxiv.org/pdf/2108.03298v1.pdf)[[Github]](https://arise-initiative.github.io/robomimic-web/)

**Towards Human-Level Bimanual Dexterous Manipulation with Reinforcement Learning**[2022]
Chen, Y. Yang, T. Wu, S. Wang, X. Feng, J. Jiang, S. M. McAleer, H. Dong, Z. Lu, and S.-C. Zhu[[PDF]](https://export.arxiv.org/pdf/2206.08686v2.pdf)[[Github]](https://github.com/PKU-MARL/DexterousHands)

**Stabilizing Voltage in Power Distribution Networks via Multi-Agent Reinforcement Learning with Transformer**[2022]
M. Wang, M. Feng, W. Zhou, and H. Li[[PDF]](https://export.arxiv.org/pdf/2206.03721.pdf)[[Github]](https://github.com/cjdjr/T-MAAC)

**Multi-Agent Reinforcement Learning for Active Voltage Control on Power Distribution Networks**[2021]
J. Wang, W. Xu, Y. Gu, W. Song, and T. C. Green[[PDF]](http://export.arxiv.org/pdf/2110.14300v4.pdf)[[Github]](https://github.com/Future-Power-Networks/MAPDN)

**Combining Reinforcement Learning and Optimal Transport for the Traveling Salesman Problem**[2022]
Y. L. Goh, W. S. Lee, X. Bresson, T. Laurent, and N. Lim[[PDF]](https://export.arxiv.org/pdf/2203.00903.pdf)

**Structure-Aware Transformer Policy for Inhomogeneous Multi-Task Reinforcement Learning**[2022]
S. Hong, D. Yoon, and K.-E. Kim[[PDF]](https://openreview.net/forum?id=fy_XRVHqly)

**Addressing Function Approximation Error in Actor-Critic Methods**[2018]
S. Fujimoto, H. van Hoof, and D. Meger[[PDF]](http://export.arxiv.org/pdf/1802.09477v3.pdf)[[Github]](https://github.com/sfujim/TD3)
 
**One Policy to Control Them All: Shared Modular Policies for Agent-Agnostic Control**[2020]
W. Huang, I. Mordatch, and D. Pathak [[PDF]](https://arxiv.org/pdf/2007.04976.pdf)[[Github]](https://huangwl18.github.io/modular-rl/)
 
**Distributed Multi-Agent Deep Reinforcement Learning for Robust Coordination against Noise**[]2022]
Y. Motokawa and T. Sugawara[[PDF]](https://export.arxiv.org/pdf/2205.09705.pdf)

**Implicit Quantile Networks for Distributional Reinforcement Learning**[2018]
W. Dabney, G. Ostrovski, D. Silver, and R. Munos[[PDF]](http://export.arxiv.org/pdf/1806.06923v1.pdf)

**Multi-Game Decision Transformers**[2022]
K.-H. Lee, O. Nachum, M. Yang, L. Lee, D. Freeman, W. Xu, S. Guadarrama, I. Fischer, E. Jang, H. Michalewski, and I. Mordatch[[PDF]](https://export.arxiv.org/pdf/2205.15241v2.pdf)

**Silver-Bullet-3D at ManiSkill 2021: Learning-from-Demonstrations and Heuristic Rule-based Methods for Object Manipulation**[2022]
Y. Pan, Y. Li, Y. Zhang, Q. Cai, F. Long, Z. Qiu, T. Yao, and T. Mei[[PDF]](https://export.arxiv.org/pdf/2206.06289.pdf)[[Github]](https://github.com/caiqi/Silver-Bullet-3D/)

**ManiSkill: Generalizable Manipulation Skill Benchmark with Large-Scale Demonstrations**[2020]
T. Mu, Z. Ling, F. Xiang, D. Yang, X. Li, S. Tao, Z. Huang, Z. Jia, and H. Su[[PDF]](https://export.arxiv.org/pdf/2107.14483.pdf)[[Github]](https://github.com/haosulab/ManiSkill)

**A learning algorithm for continually running fully recurrent neural networks**[1989]
 R. J. Williams and D. Zipser[[PDF]](https://direct.mit.edu/neco/article-abstract/1/2/270/5490/A-Learning-Algorithm-for-Continually-Running-Fully)

**SMARTS: Scalable Multi-Agent Reinforcement Learning Training School for Autonomous Driving**[2020]
M. Zhou, J. Luo, J. Villela, Y. Yang, D. Rusu, J. Miao, W. Zhang, M. Alban, I. Fadakar, Z. Chen, A. C. Huang, Y. Wen, K. Hassanzadeh, D. Graves, D. Chen, Z. Zhu, N. M. Nguyen, M. A. Elsayed, K. Shao, S. Ahilan, B. Zhang, J. Wu, Z. Fu, K. Rezaee, P. Yadmellat, M. Rohani, N. P. Nieves, Y. Ni, S. Banijamali, A. I. Cowen-Rivers, Z. Tian, D. Palenicek, H. Bou-Ammar, H. Zhang, W. Liu, J. Hao, and J. Wang[[PDF]](https://arxiv.org/pdf/2010.09776.pdf)[[Github]](https://github.com/huawei-noah/SMARTS)

**CARLA: An Open Urban Driving Simulato**[2017]
A. Dosovitskiy, G. Ros, F. Codevilla, A. M. L ́opez, and V. Koltu[[PDF]](https://arxiv.org/pdf/1711.03938.pdf)

**Catformer: Designing Stable Transformers via Sensitivity Analysis**[2021]
J. Davis, A. Gu, K. Choromanski, T. Dao, C. R ́e, C. Finn, and P. Lian[[PDF]](http://proceedings.mlr.press/v139/davis21a/davis21a.pdf)

**Recurrent Experience Replay in Distributed Reinforcement Learning**[2019]
S. Kapturowski, G. Ostrovski, J. Quan, R. Munos, and W. Dabney[[PDF]](https://openreview.net/pdf?id=r1lyTjAqYX)

**Stabilizing Transformers for Reinforcement Learning**[2020]
E. Parisotto, H. F. Song, J. W. Rae, R. Pascanu, C. Gulcehre, S. M. Jayakumar, M. Jaderberg, R. L. Kaufman, A. Clark, S. Noury, M. Botvinick, N. Heess, and R. Hadsell[[PDF]](https://export.arxiv.org/pdf/1910.06764.pdf)[[Github]](https://github.com/kimiyoung/transformer-xl)

**2048**[2014]
G. Cirulli
## Expensive Memory and Computation

**BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**[2019]
Jacob Devlin, Ming-Wei Chang, Kenton Lee, & Kristina Toutanova[[PDF]](https://arxiv.org/pdf/1810.04805.pdf)[[Github]](https://github.com/google-research/bert.)

**Unsupervised Cross-lingual Representation Learning at Scale**[2020]
A. Conneau, K. Khandelwal, N. Goyal, V. Chaudhary, G. Wenzek, F. Guzm ́an, E. Grave, M. Ott, L. Zettlemoyer, and V. Stoyanov[[PDF]](https://arxiv.org/pdf/1911.02116.pdf)[[Github]](https://github.com/facebookresearch/(fairseq-py,pytext,xlm))

**Language Models are Unsupervised Multitask Learners**[2019]
Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, & Ilya Sutskeve[[PDF]](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)[[Github]](https://github.com/codelucas/newspaper)

**DeepNet: Scaling Transformers to 1,000 Layers**[2022]
H. Wang, S. Ma, L. Dong, S. Huang, D. Zhang, and F. Wei[[PDF]](http://export.arxiv.org/pdf/2203.00555v1.pdf)[[Github]](https://github.com/microsoft/unilm)

**Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer**[2020]
Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, & Peter J. Liu[[PDF]](https://arxiv.org/pdf/1910.10683.pdf)[[Github]]( https://github.com/google-research/text-to-text-transfer-transformer)

**GLaM: Efficient Scaling of Language Models with Mixture-of-Experts**[2022]
N. Du, Y. Huang, A. M. Dai, S. Tong, D. Lepikhin, Y. Xu, M. Krikun, Y. Zhou, A. W. Yu, O. Firat, B. Zoph, L. Fedus, M. Bosma, Z. Zhou, T. Wang, Y. E. Wang, K. Webster, M. Pellat, K. Robinson, K. MeierHellstern, T. Duke, L. Dixon, K. Zhang, Q. V. Le, Y. Wu, Z. Chen, and C. Cui[[PDF]](https://export.arxiv.org/pdf/2112.06905v2.pdf)

**Scaling Laws for Neural Language Models**[2020]
J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, B. Chess, R. Child, S. Gray, A. Radford, J. Wu, and D. Amodei[[PDF]](https://arxiv.org/abs/2001.08361)

**Galvatron: Efficient Transformer Training over Multiple GPUs Using Automatic Parallelism**[2022]
X. Miao, Y. Wang, Y. Jiang, C. Shi, X. Nie, H. Zhang, and B. Cui[[PDF]](https://export.arxiv.org/pdf/2211.13878v1.pdf)

**TorchScale: Transformers at Scale**[2022]
S. Ma, H. Wang, S. Huang, W. Wang, Z. Chi, L. Dong, A. Benhaim, B. Patra, V. Chaudhary, X. Song, and F. Wei[[PDF]](https://export.arxiv.org/pdf/2211.13184v1.pdf)

**Random-LTD: Random and Layerwise Token Dropping Brings Efficient Training for Large-scale Transformers**[2022]
Z. Yao, X. Wu, C. Li, C. Holmes, M. Zhang, C. Li, and Y. He[[PDF]](https://export.arxiv.org/pdf/2211.11586v1.pdf)[[Github]](https://github.com/microsoft/DeepSpeed)


## Stochastic Effectiveness

**Addressing Optimism Bias in Sequence Modeling for Reinforcement Learning**[2022]
Adam Villaflor, Zhe Huang, Swapnil Pande, John Dolan, & Jeff Schneider[[PDF]](https://export.arxiv.org/pdf/2207.10295.pdf)[[Github]](https://github.com/avillaflor/SPLTtransformer)

**You Can't Count on Luck: Why Decision Transformers Fail in Stochastic Environments**[2022]
Keiran Paster, Sheila McIlraith, & Jimmy Ba[[PDF]](https://export.arxiv.org/pdf/2205.15967v2.pdf)[[Github]](https://github.com/PascalPons/connect4)

**Planning from Pixels using Inverse Dynamics Models**[2020]
K. Paster, S. A. McIlraith, and J. Ba[[[PDF]](http://export.arxiv.org/pdf/2012.02419)[[Github]](https://github.com/keirp/glamor)

**Vector Quantized Models for Planning**[2021]
S. Ozair, Y. Li, A. Razavi, I. Antonoglou, A. van den Oord, and O. Vinyals[[PDF]](https://arxiv.org/pdf/2106.04615.pdf)

**Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model**[2019]
J. Schrittwieser, I. Antonoglou, T. Hubert, K. Simonyan, L. Sifre, S. Schmitt, A. Guez, E. Lockhart, D. Hassabis, T. Graepel, T. P. Lillicrap, and D. Silver[[[PDF]](https://arxiv.org/pdf/1911.08265.pdf)

**Efficient Selectivity and Backup Operators in Monte-Carlo Tree Search**[2006]
R. Coulom[[PDF]](https://link.springer.com/content/pdf/10.1007/978-3-540-75538-8_7.pdf)





