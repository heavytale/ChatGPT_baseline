import asyncio
import random
import time


async def f(t):
    """ 실행에 약 t초가 소요되는 함수 """
    await asyncio.sleep(t)


async def async_wait(i):
    await asyncio.sleep(random.random() * 2)
    return i


async def main():
    results = await asyncio.gather(*[async_wait(i) for i in range(10)])
    print(results)


if __name__ == "__main__":
    asyncio.run(main())

