import pygame, sys
pygame.init()
pygame.joystick.init()
print("pygame version:", pygame.__version__)

selected = None
for i in range(pygame.joystick.get_count()):
    j = pygame.joystick.Joystick(i)
    j.init()
    name = j.get_name().lower()
    if "logitech" in name or "f310" in name:
        selected = j
        break


if selected is None:
    print("No suitable gamepad found after enumeration")
 
else:
    print(f"Selected gamepad: {selected.get_name()} with {selected.get_numaxes()} axes and {selected.get_numbuttons()} buttons")