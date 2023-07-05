1. map을 통째로 state로 넘겨주자. null, snake, food, obstacle -> 0,1,2,3
map[10][10]


mlp

X1 : B x 100
W1 : 100 x 64

X2: B x 64
W2: 64 x 64

X3 : B x 2000

차원의 저주: 모델 사이즈가 크면 그만큼 충분히 학습시키기 위해서는 많은 인풋 데이터가 필요하다.
제안: head 기준 5x5 정도만 state로 넣는 것은 차원을 충분히 감소시킬 수 있을것

2. move direction 인풋이 정말 필요한 데이터인가? noise일 수도

3. 쭉 바라보는 방향에 장애물이 얼마나 떨어져있는지. straight, right, left 3개로.

model.py 병현이가 lr_scheduler 넣어둠. 고려하삼.

