#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <conio.h>
#include <wchar.h>
#include <iostream>
#include <io.h>
#include <fcntl.h>

#define MAP_SIZE 20
#define MAX_SNAKE_LENGTH 100

int map[MAP_SIZE + 2][MAP_SIZE + 2] = { 0, };	// 빈공간: 0, 뱀: 1, 먹이: 2, 벽: 9
int snake_i[MAX_SNAKE_LENGTH + 2] = { 0, };	// 여기에 뱀의 각 부위의 i좌표 적을거임. snake_i[0]: 머리
int snake_j[MAX_SNAKE_LENGTH + 2] = { 0, };	// 뱀의 j좌표, snake_j[0]: 머리
int length = 1;	// 뱀의 길이, 먹이를 먹으면 +1
int death = 0;	// 0이면 생존, 1이면 사망
int grow = 0;	// 1이면 성장할 예정
int di, dj;

int random(int a, int b) {
	return rand()*(b - a + 1) / RAND_MAX + a;
}
void swap(int &a, int &b) {
	int t = a;
	a = b;
	b = t;
}

void print_main()
{
	_setmode(_fileno(stdout), _O_U16TEXT);
	wchar_t str[] = L"███████╗███╗   ██╗ █████╗ ██╗  ██╗███████╗     █████╗ ██╗\n██╔════╝████╗  ██║██╔══██╗██║ ██╔╝██╔════╝    ██╔══██╗██║\n███████╗██╔██╗ ██║███████║█████╔╝ █████╗      ███████║██║\n╚════██║██║╚██╗██║██╔══██║██╔═██╗ ██╔══╝      ██╔══██║██║\n███████║██║ ╚████║██║  ██║██║  ██╗███████╗    ██║  ██║██║\n╚══════╝╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝    ╚═╝  ╚═╝╚═╝\n                                                         ";
	std::wcout << str << std::endl;

	printf("===================================================================\nmove: WASD, snake: @, food: &, wall: #\n\n");
	system("pause");
}

void init_set()	// 지도 설정, 뱀 위치, 먹이 위치 랜덤 설정
{
	for (int i = 0; i <= MAP_SIZE; i++) {	//지도 설정
		map[0][i] = map[i][0] = map[MAP_SIZE][i] = map[i][MAP_SIZE] = 9;
	}	// ex) MAP_SIZE:100 -> 이동반경:1~99

	snake_i[0] = random(1, MAP_SIZE-1);	//뱀 위치 설정
	snake_j[0] = random(1, MAP_SIZE-1);
	map[snake_i[0]][snake_j[0]] = 1;

	map[random(1, MAP_SIZE-1)][random(1, MAP_SIZE-1)] = 2;	//먹이 위치 설정
}

void set_food_posit()
{
	int food_i, food_j;
	do {
		food_i = random(1, MAP_SIZE - 1);
		food_j = random(1, MAP_SIZE - 1);
		
	} while (map[food_i][food_j] != 0);

	map[food_i][food_j] = 2;
}

int input_direction()	//방향 입력후 리턴
{
	char direc;
	direc = getch();
//	scanf("%c%*c", &direc);

	switch (direc)
	{
	case 'w':
		return 3;
		break;
	case 'a':
		return 1;
		break;
	case 's':
		return 2;
		break;
	case 'd':
		return 0;
		break;
	default:
		printf("input error\n");
		return input_direction();
		break;
	}
}

void what_is_direction(int direc)
{
	switch (direc)
	{
	case 0:	//Right
		dj++;
		break;
	case 1:	//Left
		dj--;
		break;
	case 2:	//Down
		di++;
		break;
	case 3:	//Up
		di--;
		break;
	}

	if (map[snake_i[0] + di][snake_j[0] + dj] == 9 || map[snake_i[0] + di][snake_j[0] + dj] == 1) {	// 이동할 부분이 벽 or 뱀 이라면
		death = 1;
	}
	else if (map[snake_i[0] + di][snake_j[0] + dj] == 2)
	{
		length++;
		set_food_posit();
		grow = 1;
	}
}

void move()	//snake[]도 옮겨야 하고, map[][]의 뱀 정보도 옮겨야 함
{
	for (int i = length-1; i >= 0; i--) {
		snake_i[i + 1] = snake_i[i];
		snake_j[i + 1] = snake_j[i];
	}
	snake_i[0] = snake_i[1] + di;
	snake_j[0] = snake_j[1] + dj;

	map[snake_i[0]][snake_j[0]] = 1;
	
	if (!grow) {
		map[snake_i[length]][snake_j[length]] = 0;
		snake_i[length] = snake_j[length] = 0;
	}
}

void print_map()
{
	system("cls");
	for (int i = 0; i <= MAP_SIZE; i++) {
		for (int j = 0; j <= MAP_SIZE; j++) {
			if (map[i][j] == 0) printf("- ");
			else if (map[i][j] == 1) printf("@ ");
			else if (map[i][j] == 2) printf("& ");
			else if (map[i][j] == 9) printf("# ");
			else printf("%d ", map[i][j]);
		}
		printf("\n");
	}
	printf("\nsnake position: (%d, %d)\n", snake_i[0], snake_j[0]);	//snake position
	printf("snake length: %d\n", length);
}

int main()
{
	srand((int)time(NULL));
	
	print_main();
	return 0;
	init_set();

	while (1)
	{
		print_map();

		di = dj = grow = 0;
		what_is_direction(input_direction());

		if (death) {	// 사망이면 종료
			printf("death\n");
			break;
		}
		
		move();
	}

	return 0;
}

