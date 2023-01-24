# DSC180B-Project-B319-2

Create environment for loading iwildcam dataset /n
0: Open a terminal /n
1: ssh zhc023@dsmlp-login.ucsd.edu /n
2: launch-scipy-ml.sh -c 8 -m 50 -i billchen24/dsc180b-project -P Always // request 8 cpu and 50 GB, and create specific environment with my docker image /n
3: Create new terminal /n
4: ssh -N -L 8889:127.0.0.1:16585 zhc023@dsmlp-login.ucsd.edu /n
5: Go to http://localhost:8889/user/zhc023/tree/ /n
