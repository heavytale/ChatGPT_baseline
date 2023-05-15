import asyncio


class AsyncAdd:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    async def __aenter__(self):
        print('sleeping...')
        await asyncio.sleep(1.0)
        print('sleep complete')
        return self.a + self.b  # __aenter__에서 값을 반환하면 as에 지정한 변수에 들어감

    async def __aexit__(self, exc_type, exc_value, traceback):
        print("aexit")


async def main():
    async with AsyncAdd(1, 2) as result:  # async with에 클래스의 인스턴스 지정
        print(result)  # 3
        print("more sleep...")
        await asyncio.sleep(1.0)


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
loop.close()