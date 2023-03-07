# snake-ai-freeCodeCamp.org

스스로 코드 짰을 때 문제를 해결하지 못했다. 우선 유튜브의 영상을 보고 따라서 만들어보자.

https://www.youtube.com/watch?v=L8ypSXwyBds

https://github.com/patrickloeber/snake-ai-pytorch


## 게임 설계

**reward**
- eat food: +10
- game over: -10
- else: 0

**action**
- 직진: `[1, 0, 0]`
- 우회전: `[0, 1, 0]`
- 좌회전: `[0, 0, 1]`

**state**
```
[danger straight, danger right, danger left,

direction left, direction right,
direction up, direction down,

food left, food right,
food up, food down
] 
```
- 모든 value는 1 or 0
- danger: 그 방향 바로 앞에 위험한게 있는지 
- direction: 현재 진행방향, 4개 중 하나만 1 
- food: 8방향으로 나타냄 (e.g., 오른쪽아래 등)


## 보고서

- 80즈음을 넘어서 본격적인 학습이 시작됨
- 먹이 먹은 횟수가 10번을 넘어감