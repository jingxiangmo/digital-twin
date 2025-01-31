import pykos
import time
import asyncio

async def main():
    kos = pykos.KOS("10.33.10.63")
    await kos.connect()

    time_start = time.time()
    commands_sent = 0

    while True:
        await kos.actuator.command_actuators([{"actuator_id": 32, "position": 0}])
        commands_sent += 1
        if time.time() - time_start >= 1.0:
            print(f"Commands per second (CPS): {commands_sent}")
            commands_sent = 0
            time_start = time.time()

if __name__ == "__main__":
    asyncio.run(main())
