import argparse
import asyncio

from pykos import KOS


actuator_ids = [11, 12, 13, 14, 21, 22, 23, 24, 31, 32, 33, 34, 35, 41, 42, 43, 44, 45]


async def disable_torque(kos: KOS) -> None:
    for i in range(3):
        print(f"Disabling torque: {i}")
        for actuator_id in actuator_ids:
            await kos.actuator.configure_actuator(actuator_id=actuator_id, torque_enabled=False)
            await asyncio.sleep(0.1)


async def zeroing_actuators(kos: KOS) -> None:

    for actuator_id in actuator_ids:
        await kos.actuator.configure_actuator(actuator_id=actuator_id, zero_position=True)

async def home_actuators(kos: KOS) -> None:
    for actuator_id in actuator_ids:
        await kos.actuator.configure_actuator(actuator_id=actuator_id, kp=32, kd=32, torque_enabled=True)
        await kos.actuator.command_actuators([{"actuator_id": actuator_id, "position": 0}])
        await asyncio.sleep(0.1)

async def read_actuator_states(kos: KOS) -> None:
    states = await kos.actuator.get_actuators_state(actuator_ids)
    
    print("\nActuator States:")
    for state in states.states:
        print(f"Actuator {state.actuator_id}:")
        print(f"  Position: {state.position:.2f}")
        print(f"  Velocity: {state.velocity:.2f}")
        print(f"  Torque: {state.torque:.2f}")
        print(f"  Temperature: {state.temperature:.1f}°C")

async def main() -> None:
    async with KOS(ip="192.168.42.1") as kos:
        print("Disabling torque")
        await disable_torque(kos)

        print("Zeroing actuators")
        await zeroing_actuators(kos)

        print("Homing actuators")  
        await home_actuators(kos)
        
        print("\nReading actuator states")
        await read_actuator_states(kos)

        print("\nDone")


if __name__ == "__main__":
    asyncio.run(main())
