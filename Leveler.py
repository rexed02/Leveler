import os
import pygame
import numpy as np
import random
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import matplotlib.pyplot as plt
import math
from scipy.spatial import ConvexHull
import time
import pygame.gfxdraw
from OpenGL.GL import glBegin, glEnd, glVertex2f, GL_LINES, glLineWidth, glColor3f
import cv2




# Configuración inicial
pygame.init()
pygame.mixer.init()  # Inicializar el módulo de sonido
pygame.display.set_caption('Leveler')
clock = pygame.time.Clock()
programIcon = pygame.image.load('icon.ico')
pygame.display.set_icon(programIcon)
font_path = os.path.join('Switzer-Medium.otf')
font_path_bold = os.path.join('Switzer-Bold.otf')

# Cargar los archivos de sonido
good_sound = pygame.mixer.Sound('good.mp3')
bad_sound = pygame.mixer.Sound('bad.mp3')

# Tamaño de la ventana y configuración DPI
WINDOW_SIZE = (800, 600)
pygame.display.set_mode(WINDOW_SIZE, DOUBLEBUF | OPENGL)

gluPerspective(45, (WINDOW_SIZE[0] / WINDOW_SIZE[1]), 0.1, 50.0)
glTranslatef(0.0, 0.0, -5)
screen = None 

#Cache
cube_cache_surface = None
sphere_cache_surface = None
cache_dirty = {'cube': True, 'sphere': True}



# Ocultar el cursor del ratón
pygame.mouse.set_visible(False)

# Colores
WHITE = (248, 240, 238)
RED = (200, 43, 67)
GREEN = (71, 215, 122)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GRAY = (128, 128, 128)
Amarillo = (122,81,180)
Violeta = (115,106,150)

BACKGROUND = (244, 244, 244)  
TEXT = (227, 229, 233)  # Texto claro
PRIMARY = (71, 215, 172)  # Verde neón
SECONDARY = (200, 43, 167)  # Magenta
ACCENT = (101, 101, 241)  # Azul neón
GRAPH_BACKGROUND = (255, 233, 125)  # Amarillo para el fondo del gráfico

# Crear un cubo en el espacio 3D
cube_vertices = np.array([
    [-1, -1, -1],
    [1, -1, -1],
    [1, 1, -1],
    [-1, 1, -1],
    [-1, -1, 1],
    [1, -1, 1],
    [1, 1, 1],
    [-1, 1, 1]
])

# Definir las caras del cubo (cada cara es un grupo de cuatro vértices)
cube_faces = [
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [0, 1, 5, 4],
    [2, 3, 7, 6],
    [0, 3, 7, 4],
    [1, 2, 6, 5]
]

# Opciones de tamaño para cubo y esfera
size_options = {
    'small': 1,
    'medium': 1.5,
    'large': 2
}
current_size = 'medium'

# Variables para la esfera
current_shape = 'cube'  # 'cube' o 'sphere'
sphere_radius = size_options[current_size]*2
sphere_center = np.array([0, 0, 0])
sphere_ellipses = []
visible_ellipse = None


# Variables para el rombo
rhombus_size = 24  # Tamaño base del rombo
rhombus_x = 40  # Posición base en X
rhombus_y = 25  # Posición base en Y
rhombus_color = (115, 106, 150)  # Color del rombo

def draw_rhombus(screen, size, x, y, color):
    screen_width, screen_height = screen.get_size()
    scaled_size = int(size * screen_height / 600)
    scaled_x = int(x * screen_height / 600)
    scaled_y = int(y * screen_height / 600)
    
    points = [
        (scaled_x, scaled_y - scaled_size),  # Punto superior
        (scaled_x + scaled_size, scaled_y),  # Punto derecho
        (scaled_x, scaled_y + scaled_size),  # Punto inferior
        (scaled_x - scaled_size, scaled_y)   # Punto izquierdo
    ]
    
    pygame.draw.polygon(screen, color, points)


def draw_axes(screen, center, size, camera_position, screen_size, fov, viewer_distance):
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Rojo, Verde, Azul
    directions = [
        np.array([size, 0, 0]),
        np.array([0, size, 0]),
        np.array([0, 0, size])
    ]
    
    for i, direction in enumerate(directions):
        start = center
        end = center + direction
        start_deformed = fish_eye_deformation(start)
        end_deformed = fish_eye_deformation(end)
        screen_start = project(start_deformed, camera_position, screen_size, fov, viewer_distance)
        screen_end = project(end_deformed, camera_position, screen_size, fov, viewer_distance)
        pygame.draw.line(screen, colors[i], screen_start, screen_end, 2)



def draw_thick_line(screen, color, start, end, thickness):
    integer_thickness = int(thickness)
    decimal_part = thickness - integer_thickness
    
    # Dibuja la línea principal con el grosor entero
    points = draw_single_thick_line(screen, color, start, end, integer_thickness)
    
    # Dibuja las líneas aaline para los bordes interior y exterior
    
    pygame.draw.line(screen, color, points[0], points[1],2)  # Borde superior
    pygame.draw.line(screen, color, points[3], points[2],3)  # Borde inferior
    
    # Si hay una parte decimal, dibuja una línea adicional más delgada usando aaline
    if decimal_part > 0:
        alpha = int(decimal_part * 255)
        additional_color = color[:3] + (alpha,)  # Añade canal alpha
        pygame.draw.aaline(screen, color, start, end)

def draw_single_thick_line(screen, color, start, end, thickness):
    center = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
    angle = math.atan2(start[1] - end[1], start[0] - end[0])
    length = math.hypot(end[0] - start[0], end[1] - start[1])

    UL = (center[0] + (length / 2) * math.cos(angle) - (thickness / 2) * math.sin(angle),
          center[1] + (thickness / 2) * math.cos(angle) + (length / 2) * math.sin(angle))
    UR = (center[0] - (length / 2) * math.cos(angle) - (thickness / 2) * math.sin(angle),
          center[1] + (thickness / 2) * math.cos(angle) - (length / 2) * math.sin(angle))
    BL = (center[0] + (length / 2) * math.cos(angle) + (thickness / 2) * math.sin(angle),
          center[1] - (thickness / 2) * math.cos(angle) + (length / 2) * math.sin(angle))
    BR = (center[0] - (length / 2) * math.cos(angle) + (thickness / 2) * math.sin(angle),
          center[1] - (thickness / 2) * math.cos(angle) - (length / 2) * math.sin(angle))

    pygame.gfxdraw.aapolygon(screen, (UL, UR, BR, BL), color)
    pygame.gfxdraw.filled_polygon(screen, (UL, UR, BR, BL), color)

    return (UL, UR, BR, BL)




def position_camera_sphere():
    # Mantener un radio constante para la distancia en el plano XY
    radius_xy = random.uniform(4, 6)  # Variar ligeramente el radio pero mantenerlo cerca
    angle_xy = random.uniform(0, 2 * np.pi)
    
    # Variar la profundidad (coordenada Z) para que siempre esté en un rango visible
    depth = random.uniform(-2, 4.5)  # Ajusta estos valores según necesites
    
    base_position = np.array([
        radius_xy * np.cos(angle_xy),
        radius_xy * np.sin(angle_xy),
        depth
    ])
    
    # Agregar un pequeño offset aleatorio para más variación
    offset = np.array([
        random.uniform(-0.5, 0.5),
        random.uniform(-0.5, 0.5),
        random.uniform(-0.2, 0.2)  # Menor variación en Z para no contrarrestar demasiado el efecto de profundidad
    ])
    
    return base_position + offset





# Función para crear o recrear la ventana
def crear_ventana(tamaño):
    global screen
    try:
        if screen is None or screen.get_size() != tamaño:
            screen = pygame.display.set_mode(tamaño, RESIZABLE | HWSURFACE | DOUBLEBUF)
    except pygame.error:
        print("No se pudo crear la ventana con aceleración por hardware. Cambiando a renderizado por software.")
        screen = pygame.display.set_mode(tamaño, RESIZABLE | SWSURFACE)
    return screen

screen = crear_ventana(WINDOW_SIZE)

def create_rotated_rect(center, width, height, angle):
    angle_rad = math.radians(angle)
    half_width, half_height = width / 2, height / 2

    points = [
        (half_width, half_height),
        (-half_width, half_height),
        (-half_width, -half_height),
        (half_width, -half_height)
    ]

    rotated_points = []
    for x, y in points:
        new_x = x * math.cos(angle_rad) - y * math.sin(angle_rad)
        new_y = x * math.sin(angle_rad) + y * math.cos(angle_rad)
        rotated_points.append((center[0] + new_x, center[1] + new_y))

    return rotated_points

import pygame
import math

import pygame

def draw_shape_buttons(screen, corner_cut_percentage=0.25, show_clickable_area=False):
    screen_width, screen_height = screen.get_size()
    button_size = int(40 * screen_height / 600)
    margin = int(20 * screen_height / 600)
    outline_thickness = int(2 * screen_height / 600)
    corner_cut = int(button_size * corner_cut_percentage)
    inner_symbol_thickness = int(2 * screen_height / 600)
    
    def draw_button(pos, is_cube):
        x, y = pos
        points = [
            (x + corner_cut, y),
            (x + button_size, y),
            (x + button_size, y + button_size - corner_cut),
            (x + button_size - corner_cut, y + button_size),
            (x, y + button_size),
            (x, y + corner_cut)
        ]
        
        # Draw yellow fill
        pygame.draw.polygon(screen, (242, 230, 230), points)
        
        # Draw button outline
        pygame.draw.lines(screen, (3, 4, 0), True, points, outline_thickness)
        
        # Draw inner shape (square for cube, circle for sphere)
        inner_size = int(button_size * 0.25)
        inner_pos = (x + (button_size - inner_size) // 2, y + (button_size - inner_size) // 2)
        if is_cube:
            inner_pos = (x + (button_size - inner_size+2) // 2, y + (button_size - inner_size+2) // 2)
            pygame.draw.rect(screen, (3, 4, 0), (*inner_pos, inner_size-1, inner_size-1), inner_symbol_thickness)
        else:
            pygame.draw.circle(screen, (3, 4, 0), (x + button_size // 2, y + button_size // 2), 
                               inner_size // 2, inner_symbol_thickness)
        
        return points

    cube_pos = (margin, margin)
    sphere_pos = (margin, margin/2 + button_size + margin)

    cube_clickable = draw_button(cube_pos, True)
    sphere_clickable = draw_button(sphere_pos, False)

    if show_clickable_area:
        pygame.draw.polygon(screen, (0, 255, 0), cube_clickable, 2)
        pygame.draw.polygon(screen, (0, 255, 0), sphere_clickable, 2)

    return cube_clickable, sphere_clickable



def draw_trapezoid_aa(surface, color, points, line_width):
    # Convertir puntos a enteros
    int_points = [(int(x), int(y)) for x, y in points]
    # Dibujar el trapecio con antialiasing
    pygame.draw.aalines(surface, color, True, int_points, line_width)

def rotate_point(point, angle, origin):
    ox, oy = origin
    px, py = point
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return (qx, qy)

def create_equilateral_trapezoid(center, width, height):
    half_width = width / 2
    half_height = height / 2
    top_width = width * 0.6  # Ajusta este valor para cambiar la forma del trapecio
    half_top_width = top_width / 2
    
    return [
        (center[0] - half_top_width, center[1] - half_height),
        (center[0] + half_top_width, center[1] - half_height),
        (center[0] + half_width, center[1] + half_height),
        (center[0] - half_width, center[1] + half_height)
    ]



def draw_difficulty_size_reset(screen):
    screen_width, screen_height = screen.get_size()
    font_size = int(16 * screen_height / 600)  # Escalar tamaño de fuente
    font = pygame.font.Font(font_path, font_size)
    text = font.render(f"D:{difficulty_levels[current_difficulty]}      T:{size_options[current_size]}", True, (45,43,44))
    
    bg_rect = pygame.Rect(-10, screen_height - int(142 * screen_height / 600), int(190 * screen_height / 600), int(35 * screen_height / 600))
    #pygame.draw.rect(screen, (63,50,65), bg_rect, border_radius=int(7 * screen_height / 600))
    
    screen.blit(text, (18 * screen_height / 600, screen_height - int(136 * screen_height / 600)))
    
    reset_image = pygame.image.load('flecha.png')
    reset_image = pygame.transform.scale(reset_image, (int(20 * screen_height / 600), int(20 * screen_height / 600)))
    reset_rect = reset_image.get_rect(midleft=(bg_rect.right - int(60 * screen_height / 600), bg_rect.centery))
    #pygame.draw.rect(screen, (63,50,65), reset_rect.inflate(int(10 * screen_height / 600), int(10 * screen_height / 600)), border_radius=int(10 * screen_height / 600))
    screen.blit(reset_image, reset_rect)

    # Parámetros para los trapecios
    trapezoid_scale = 0.38
    base_width = int(50 * screen_height / 600)
    base_height = int(15 * screen_height / 600)
    trapezoid_width = int(base_width * trapezoid_scale)
    trapezoid_height = int(base_height * trapezoid_scale)
    trapezoid_offset = int(28 * screen_height / 600)
    trapezoid_offset2 = int(2 * screen_height / 600)
    line_width = 2
    color = (45, 43, 44)
    
    # Primer trapecio
    trap1_center = (reset_rect.right + trapezoid_offset, reset_rect.centery+trapezoid_offset2)
    trap1_points = create_equilateral_trapezoid(trap1_center, trapezoid_width, trapezoid_height)
    
    # Rotar el primer trapecio (125 grados)
    rotated_trap1_points = [rotate_point(p, math.radians(125), trap1_center) for p in trap1_points]
    draw_trapezoid_aa(screen, color, rotated_trap1_points, line_width)

    # Segundo trapecio
    trapezoid_offset = int(40 * screen_height / 600)
    trap2_center = (reset_rect.right + trapezoid_offset, reset_rect.centery+trapezoid_offset2)
    trap2_points = create_equilateral_trapezoid(trap2_center, trapezoid_width, trapezoid_height)
    
    # Rotar el segundo trapecio (125 grados)
    rotated_trap2_points = [rotate_point(p, math.radians(125), trap2_center) for p in trap2_points]
    draw_trapezoid_aa(screen, color, rotated_trap2_points, line_width)



    
    return reset_rect  # Retorna el rectángulo del botón de reset para detección de clics

def draw_reset_button(screen):
    reset_image = pygame.image.load('flecha.png')
    reset_image = pygame.transform.scale(reset_image, (30, 30))
    screen.blit(reset_image, (150, screen.get_height() - 35))

def draw_accuracy(screen):
    screen_width, screen_height = screen.get_size()
    font_size = int(27 * screen_height / 600)
    font = pygame.font.Font(font_path_bold, font_size)
    text = font.render(f"{accuracy:.2f}%", True, (4,0,0))
    
    rect_width = int(140 * screen_height / 600)
    rect_height = int(50 * screen_height / 600)
    x = screen_width - rect_width - int(1 * screen_height / 600)
    y = screen_height - rect_height - int(50 * screen_height / 600)
    
    corner_cut_percentage = 0.3
    corner_cut = int(min(rect_width, rect_height) * corner_cut_percentage)
    
    points = [
        (x, y),  # Esquina superior izquierda
        (x + rect_width - corner_cut, y),  # Esquina superior derecha (cortada)
        (x + rect_width, y + corner_cut),
        (x + rect_width, y + rect_height),  # Esquina inferior derecha
        (x + corner_cut, y + rect_height),  # Esquina inferior izquierda (cortada)
        (x, y + rect_height - corner_cut)
    ]
    
    # Draw background fill
    pygame.draw.polygon(screen, (242,230,230), points)
    
    # Draw outline
    pygame.draw.lines(screen, (3, 4, 0), True, points, 3)
    
    # Position and draw text
    text_rect = text.get_rect(center=(x + rect_width // 2, y + rect_height // 2))
    screen.blit(text, text_rect)

def draw_graph(screen, cut_size=35):
    screen_width, screen_height = screen.get_size()
    relative_cut_size = int(cut_size * screen_height / 600)
    graph_rect1 = pygame.Rect(0, screen_height - int(101 * screen_height / 600), int(276 * screen_height / 600), int(101 * screen_height / 600))
    graph_rect = pygame.Rect(0, screen_height - int(100 * screen_height / 600), int(275 * screen_height / 600), int(100 * screen_height / 600))

    # Create a polygon for the graph background with the diagonal cut
    cut_points = [
        (graph_rect.left, graph_rect.top),
        (graph_rect.right - relative_cut_size, graph_rect.top),
        (graph_rect.right, graph_rect.top + relative_cut_size),
        (graph_rect.right, graph_rect.bottom),
        (graph_rect.left, graph_rect.bottom)
    ]

    # Draw the background
    pygame.draw.polygon(screen, (106, 92, 72), cut_points)
    
    # Draw the inner part
    inner_cut_points = [
        (graph_rect.left + 1, graph_rect.top + 1),
        (graph_rect.right - relative_cut_size - 1, graph_rect.top + 1),
        (graph_rect.right - 1, graph_rect.top + relative_cut_size + 1),
        (graph_rect.right - 1, graph_rect.bottom - 1),
        (graph_rect.left + 1, graph_rect.bottom - 1)
    ]
    pygame.draw.polygon(screen, (48, 46, 47), inner_cut_points)

    if len(accuracy_history) > 1:
        max_accuracy = max(accuracy_history) or 1  # Use 1 if max_accuracy is 0
        points = []
        for i, acc in enumerate(accuracy_history):
            x = graph_rect.left + i * (graph_rect.width / (len(accuracy_history) - 1))
            y = graph_rect.bottom - (acc / max_accuracy) * graph_rect.height

            # Ensure points do not exceed the cut-off limit
            if x > graph_rect.right - relative_cut_size:
                # Calculate the intersection point with the cut edge
                m = (graph_rect.top + relative_cut_size - graph_rect.top) / relative_cut_size
                y = max(y, graph_rect.top + m * (x - (graph_rect.right - relative_cut_size)))

            points.append((x, y))
        
        # Area under the curve
        points_for_fill = points + [(graph_rect.right, graph_rect.bottom), (graph_rect.left, graph_rect.bottom)]
        pygame.draw.polygon(screen, (*(61, 59, 60), 100), points_for_fill)
        
        # Thicker line
        pygame.draw.lines(screen, (131, 131, 131), False, points, int(4.5 * screen_height / 600))




def interpolate_points(p1, p2, num_points=20):
    return [(p1[0] + (p2[0] - p1[0]) * t / num_points,
             p1[1] + (p2[1] - p1[1]) * t / num_points,
             p1[2] + (p2[2] - p1[2]) * t / num_points) for t in range(num_points + 1)]



def draw_grid(screen, camera_position, screen_size, fov, viewer_distance, grid_size=10, step=2):
    for x in range(-grid_size, grid_size + 1, step):
        for y in range(-grid_size, grid_size + 1, step):
            p1 = interpolate_points((x, y, -grid_size), (x, y, grid_size))
            p2 = interpolate_points((x, -grid_size, y), (x, grid_size, y))
            p3 = interpolate_points((-grid_size, x, y), (grid_size, x, y))
            
            for segment in [p1, p2, p3]:
                for i in range(len(segment) - 1):
                    start = fish_eye_deformation(segment[i])
                    end = fish_eye_deformation(segment[i + 1])
                    screen_start = project(start, camera_position, screen_size, fov, viewer_distance)
                    screen_end = project(end, camera_position, screen_size, fov, viewer_distance)
                    pygame.draw.line(screen, GRAY, screen_start, screen_end)

def fish_eye_deformation(vertex, intensity=0.09):
    x, y, z = vertex
    distance = np.sqrt(x**2 + y**2 + z**2)
    factor = 1 / (1 + intensity * distance)
    return x * factor, y * factor, z * factor


# Función para escalar los vértices del cubo según el tamaño seleccionado
def scale_cube_vertices(scale):
    return cube_vertices * scale

scaled_cube_vertices = scale_cube_vertices(size_options[current_size])

# Función para proyectar vértices 3D a 2D
def project(vertex, camera_position, screen_size, fov, viewer_distance):
    # Convertir el vértice y la posición de la cámara a arrays numpy
    vertex = np.array(vertex)
    camera_position = np.array(camera_position)
    
    # Calcular el vector desde la cámara al vértice
    view_vector = vertex - camera_position
    
    # Proyectar el punto
    factor = fov / (viewer_distance + view_vector[2])
    x = view_vector[0] * factor + screen_size[0] // 2
    y = -view_vector[1] * factor + screen_size[1] // 2
    return int(x), int(y)
# Posicionar la cámara aleatoriamente alrededor del objeto, manteniendo el objeto en vista
def position_camera():
    radius = random.uniform(0, 4)
    angle = random.uniform(0, 2 * np.pi)
    return np.array([radius * np.cos(angle), radius * np.sin(angle), 0])

camera_pos = position_camera()

# Función para rotar un punto alrededor de un eje
def rotate(point, angle, axis):
    sin_angle = np.sin(angle)
    cos_angle = np.cos(angle)
    if axis == 'x':
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, cos_angle, -sin_angle],
            [0, sin_angle, cos_angle]
        ])
    elif axis == 'y':
        rotation_matrix = np.array([
            [cos_angle, 0, sin_angle],
            [0, 1, 0],
            [-sin_angle, 0, cos_angle]
        ])
    elif axis == 'z':
        rotation_matrix = np.array([
            [cos_angle, -sin_angle, 0],
            [sin_angle, cos_angle, 0],
            [0, 0, 1]
        ])
    return np.dot(point, rotation_matrix)

# Aplicar una rotación inicial aleatoria al cubo
def apply_initial_rotation():
    initial_rotation = {
        'x': random.uniform(0, 2 * np.pi),
        'y': random.uniform(0, 2 * np.pi),
        'z': random.uniform(0, 2 * np.pi)
    }
    rotated_vertices = [rotate(v, initial_rotation['x'], 'x') for v in scaled_cube_vertices]
    rotated_vertices = [rotate(v, initial_rotation['y'], 'y') for v in rotated_vertices]
    rotated_vertices = [rotate(v, initial_rotation['z'], 'z') for v in rotated_vertices]
    return rotated_vertices

rotated_vertices = apply_initial_rotation()

# Seleccionar una cara aleatoria para ser visible
def select_visible_face():
    return random.choice(cube_faces)

visible_face = select_visible_face()

# Variable para alternar la visibilidad de las líneas negras
show_lines = False

# Variable para almacenar el estado del programa (reset)
reset_requested = False

# Variable para el modo de dibujo (pincel o goma)
drawing_mode = 'pencil'  # 'pencil' o 'eraser'
brush_size = 2

# Almacenar el historial de precisión
accuracy_history = []
vuelta = []
vv = 0

# Definir el grosor de las líneas del cubo en función de la dificultad
difficulty_levels = {
    'dificil': 3,
    'normal': 2,
    'facil': 1
}
current_difficulty = 'normal'
line_thickness = difficulty_levels[current_difficulty]

# Función para generar puntos de una elipse en 3D
def generate_3d_ellipse(center, normal, radius, height, num_points=25):  # Cambiar de 100 a 200 o más
    u = np.cross(normal, [1, 0, 0])
    if np.all(u == 0):
        u = np.cross(normal, [0, 1, 0])
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    
    theta = np.linspace(0, 2*np.pi, num_points)
    points = center + radius * np.cos(theta)[:, np.newaxis] * u + radius * np.sin(theta)[:, np.newaxis] * v
    return points


# Función para generar tres elipses perpendiculares en la esfera
def generate_sphere_ellipses():

    global sphere_ellipses, camera_pos, sphere_center
    
    # Generar normal1 aleatoriamente
    normal1 = np.random.rand(3) - 0.5
    normal1 /= np.linalg.norm(normal1)
    
    # Generar normal2 perpendicular a normal1
    normal2 = np.cross(normal1, np.random.rand(3) - 0.5)
    normal2 /= np.linalg.norm(normal2)
    
    # Cantidad de elipses
    cantidad_de_elipses = 20  # Variable para modificar la cantidad de elipses
  
    
    # Generar normales adicionales para elipses con distancias relativas constantes

    mul1=np.cross(normal1, normal2)
    mul2=np.cross(normal2, normal1)

    normals_x = []
    for i in range(cantidad_de_elipses):
        angle = i * (2 * np.pi / cantidad_de_elipses)
        normal = np.cos(angle) * normal1 + np.sin(angle) * mul1
        normal /= np.linalg.norm(normal)
        normals_x.append(normal)
    
    normals_y = []
    for i in range(cantidad_de_elipses):
        angle = i * (2 * np.pi / cantidad_de_elipses)
        normal = np.cos(angle) * normal2 + np.sin(angle) * mul2
        normal /= np.linalg.norm(normal)
        normals_y.append(normal)
    
    sphere_ellipses = []
    # Generar las primeras dos elipses
    first_ellipse = generate_3d_ellipse(sphere_center, normal1, sphere_radius, sphere_radius)
    second_ellipse = generate_3d_ellipse(sphere_center, normal2, sphere_radius, sphere_radius)
    sphere_ellipses.append(first_ellipse)
    sphere_ellipses.append(second_ellipse)

    
    # Generar y añadir elipses adicionales a partir de normals_x
    for i, normal in enumerate(normals_x):
      
        ellipse = generate_3d_ellipse(sphere_center, normal, sphere_radius, sphere_radius)
        sphere_ellipses.append(ellipse)

    
    # Generar y añadir elipses adicionales a partir de normals_y
    for i, normal in enumerate(normals_y):

        ellipse = generate_3d_ellipse(sphere_center, normal, sphere_radius, sphere_radius)
        sphere_ellipses.append(ellipse)

    








# Dibujar el cubo
def draw_cube(show_lines, screen_size):
    global scaled_cube_vertices, camera_pos
    for face in cube_faces:
        face_vertices = [rotated_vertices[i] for i in face]
        segments = [interpolate_points(face_vertices[i], face_vertices[(i + 1) % 4]) for i in range(4)]
        
        for segment in segments:
            for i in range(len(segment) - 1):
                start = fish_eye_deformation(segment[i])
                end = fish_eye_deformation(segment[i + 1])
                screen_start = project(start, camera_pos, screen_size, 256, 5)
                screen_end = project(end, camera_pos, screen_size, 256, 5)
                
                if face == visible_face:
                    color = Violeta if np.dot((np.mean(face_vertices, axis=0) - camera_pos), [0, 0, 1]) > 0 else Amarillo
                    draw_thick_line(screen, color, screen_start, screen_end, line_thickness-1.95)
                    pygame.draw.line(screen, GREEN, project(np.mean(face_vertices, axis=0), camera_pos, screen_size, 256, 5), project(camera_pos, camera_pos, screen_size, 256, 5), 2)
                else:
                    if show_lines:
                        draw_thick_line(screen, BLACK, screen_start, screen_end, line_thickness-1.95)
    


#Bandera para mostrar u ocultar líneas de la esfera
show_sphere_lines = False
all_points = []

# Función para dibujar la esfera
def draw_sphere(show_lines, screen_size):
    global camera_pos, sphere_center, all_points
    # Definir el color naranja para el borde
    ORANGE = (255, 165, 0)
    
    # Crear una lista para almacenar todos los puntos proyectados de la esfera
    all_points = []    
    colors = [Violeta, Amarillo, BLACK]
    for i, ellipse in enumerate(sphere_ellipses):
        segments = [interpolate_points(ellipse[j], ellipse[(j + 1) % len(ellipse)]) for j in range(len(ellipse))]
        
        for segment in segments:
            for j in range(len(segment) - 1):
                start = fish_eye_deformation(segment[j])
                end = fish_eye_deformation(segment[j + 1])
                screen_start = project(start, camera_pos, screen_size, 256, 5)
                screen_end = project(end, camera_pos, screen_size, 256, 5)
                
                # Añadir puntos a la lista
                all_points.append(screen_start)
                all_points.append(screen_end)

                if show_lines and i < 2:
                    draw_thick_line(screen, colors[i], screen_start, screen_end, line_thickness-1)
                    continue

                if i < 42 and i > 1 and show_sphere_lines:  # Para las elipses violeta y amarilla
                    pygame.draw.line(screen, BLACK, screen_start, screen_end, 2)
                else:
                    break

        if i < 2:  # Dibujar líneas adicionales para las elipses violeta y amarilla
            mid_point = len(ellipse) // 2
            quarter_point = mid_point // 2
            three_quarters_point = mid_point + quarter_point

            start = fish_eye_deformation(ellipse[0])
            end = fish_eye_deformation(ellipse[mid_point])
            screen_start = project(start, camera_pos, screen_size, 256, 5)
            screen_end = project(end, camera_pos, screen_size, 256, 5)
            draw_thick_line(screen, colors[i], screen_start, screen_end, line_thickness - 2)
            
            start = fish_eye_deformation(ellipse[quarter_point])
            end = fish_eye_deformation(ellipse[three_quarters_point])
            screen_start = project(start, camera_pos, screen_size, 256, 5)
            screen_end = project(end, camera_pos, screen_size, 256, 5)
            draw_thick_line(screen, colors[i], screen_start, screen_end, line_thickness - 2)

    if len(all_points) > 3 and show_lines:  # Necesitamos al menos 3 puntos para crear un contorno
        hull = ConvexHull(all_points)
        # Dibujar el borde negro con grosor reducido
        for simplex in hull.simplices:
            draw_thick_line(screen, BLACK, all_points[simplex[0]], all_points[simplex[1]], line_thickness-1)

    # Dibujar la línea roja desde el centro de la esfera hacia la cámara
    sphere_center_screen = project(sphere_center, camera_pos, screen_size, 256, 5)
    camera_screen = project(camera_pos, camera_pos, screen_size, 256, 5)
    pygame.draw.line(screen, RED, sphere_center_screen, camera_screen, 2)

    # Dibujar un punto en el centro de la pantalla para referencia
    center_screen = (screen_size[0] // 2, screen_size[1] // 2)
    pygame.draw.circle(screen, GREEN, center_screen, 5)



# Función para dibujar la forma actual (cubo o esfera)
def draw_shape(show_lines, screen_size):
    screen.fill(WHITE)
    if current_shape == 'cube':
        draw_cube(show_lines, screen_size)
    else:
        draw_sphere(show_lines, screen_size)

# Generar los puntos de las líneas del cubo, teniendo en cuenta el grosor de la línea
def generate_cube_points(screen_size):
    points = set()

    for face in cube_faces:
        face_vertices = [rotated_vertices[i] for i in face]
        segments = [interpolate_points(face_vertices[i], face_vertices[(i + 1) % 4]) for i in range(4)]
        
        for segment in segments:
            for i in range(len(segment) - 1):
                start = fish_eye_deformation(segment[i])
                end = fish_eye_deformation(segment[i + 1])
                screen_start = project(start, camera_pos, screen_size, 256, 5)
                screen_end = project(end, camera_pos, screen_size, 256, 5)
                
                # Genera puntos a lo largo de la línea proyectada
                num_points = 20  # Ajusta este número según sea necesario
                for t in np.linspace(0, 1, num_points):
                    x = int(screen_start[0] + t * (screen_end[0] - screen_start[0]))
                    y = int(screen_start[1] + t * (screen_end[1] - screen_start[1]))
                    
                    # Añade puntos alrededor para el grosor de la línea
                    for dx in range(-line_thickness, line_thickness + 1):
                        for dy in range(-line_thickness, line_thickness + 1):
                            if dx*dx + dy*dy <= line_thickness*line_thickness:
                                points.add((x + dx, y + dy))

    return points

# Función para generar puntos de la esfera
def generate_sphere_points(screen_size):
    global all_points
    points = set()
    contador_de_elipses = 0

    for ellipse in sphere_ellipses:
        segments = [interpolate_points(ellipse[j], ellipse[(j + 1) % len(ellipse)]) for j in range(len(ellipse))]

        #Esto se lo agruegue yo hehe
        contador_de_elipses += 1
        if contador_de_elipses == 3:
            break


        for segment in segments:
            for i in range(len(segment) - 1):
                start = fish_eye_deformation(segment[i])
                end = fish_eye_deformation(segment[i + 1])
                screen_start = project(start, camera_pos, screen_size, 256, 5)
                screen_end = project(end, camera_pos, screen_size, 256, 5)
                
                num_points = 20  # Ajusta según sea necesario
                for t in np.linspace(0, 1, num_points):
                    x = int(screen_start[0] + t * (screen_end[0] - screen_start[0]))
                    y = int(screen_start[1] + t * (screen_end[1] - screen_start[1]))
                    
                    for dx in range(-line_thickness, line_thickness + 1):
                        for dy in range(-line_thickness, line_thickness + 1):
                            if dx*dx + dy*dy <= line_thickness*line_thickness:
                                points.add((x + dx, y + dy))

    # Generar el contorno de la esfera
    
    if len(all_points) > 3:  # Necesitamos al menos 3 puntos para crear un contorno
        hull = ConvexHull(all_points)
        for simplex in hull.simplices:
            start = all_points[simplex[0]]
            end = all_points[simplex[1]]
            for t in np.linspace(0, 1, 100):  # Aumentar el número de puntos para mayor precisión
                x = int(start[0] + t * (end[0] - start[0]))
                y = int(start[1] + t * (end[1] - start[1]))
                for dx in range(-line_thickness, line_thickness + 1):
                    for dy in range(-line_thickness, line_thickness + 1):
                        if dx * dx + dy * dy <= line_thickness * line_thickness:
                            points.add((x + dx, y + dy))
                            all_points.append((x + dx, y + dy))





    return points

# Función para generar puntos de la forma actual (cubo o esfera)
def generate_shape_points(screen_size):
    if current_shape == 'cube':
        return generate_cube_points(screen_size)
    else:
        return generate_sphere_points(screen_size)

# Algoritmo de Bresenham para generar puntos entre dos puntos, teniendo en cuenta el grosor de la línea
def bresenham(start, end, thickness):
    x1, y1 = start
    x2, y2 = end
    points = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    while True:
        points.extend(expand_thickness((x1, y1), thickness))
        if x1 == x2 and y1 == y2:
            break
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy
    return points

# Función para expandir el grosor de un punto en un círculo
def expand_thickness(point, thickness):
    px, py = point
    expanded_points = []
    for x in range(px - thickness, px + thickness + 1):
        for y in range(py - thickness, py + thickness + 1):
            if (x - px) ** 2 + (y - py) ** 2 <= thickness ** 2:
                expanded_points.append((x, y))
    return expanded_points

# Función para calcular el porcentaje de precisión del dibujo del usuario
def calculate_accuracy(user_lines, shape_points, screen_size):
    user_points = set()
    for line in user_lines:
        for i in range(len(line['points']) - 1):
            user_points.update(bresenham(line['points'][i], line['points'][i + 1], line['brush_size']))
    intersection = user_points.intersection(shape_points)
    return len(intersection) / len(shape_points) * 100 if shape_points else 0

# Reiniciar la aplicación
def reset_application():
    global vv, plt, vuelta, rotated_vertices, visible_face, camera_pos, user_lines, accuracy, show_lines, drawing_mode, brush_size, sphere_ellipses, visible_ellipse, sphere_center
    global cache_dirty
    if current_shape == 'cube':
        rotated_vertices = apply_initial_rotation()
        visible_face = select_visible_face()
        camera_pos = position_camera()
        cache_dirty['cube'] = True
    else:
        generate_sphere_ellipses()
        camera_pos = position_camera()
        sphere_center = np.array([
            random.uniform(-0.5, 0.5),
            random.uniform(-0.5, 0.5),
            random.uniform(-0.2, 0.2)
        ])
        cache_dirty['sphere'] = True
    visible_ellipse = random.randint(0, 2)
    user_lines = []
    if accuracy_history:
        previous_accuracy = accuracy_history[-1]
        if accuracy > previous_accuracy:
            good_sound.play()
        elif accuracy < previous_accuracy:
            bad_sound.play()
    accuracy_history.append(accuracy)
    vuelta.append(vv + 1)
    plt.plot(vuelta, accuracy_history)
    plt.xlabel('Tries')
    plt.ylabel('Accuracy')
    plt.title('Resultados en el tiempo')
    vv = vv + 1
    accuracy = 0
    show_lines = False
    drawing_mode = 'pencil'
    brush_size = 2




# Dibujar la barra de selección del grosor del pincel
def draw_brush_size_selector(screen_size):
    font = pygame.font.SysFont(None, int(screen_size[1] * 0.04))
    text = font.render('Brush Size:', True, BLACK)
    screen.blit(text, (int(screen_size[0] * 0.01), int(screen_size[1] - screen_size[1] * 0.07)))
    for i in range(1, 6):
        color = RED if i == brush_size else BLACK
        pygame.draw.circle(screen, color, (int(screen_size[0] * 0.1 + i * 20), int(screen_size[1] - screen_size[1] * 0.05)), i)
        pygame.draw.circle(screen, WHITE, (int(screen_size[0] * 0.1 + i * 20), int(screen_size[1] - screen_size[1] * 0.05)), i, 1)

# Cambiar la dificultad del juego
def change_difficulty(level):
    global current_difficulty, line_thickness
    current_difficulty = level
    line_thickness = difficulty_levels[current_difficulty]

# Cambiar el tamaño del objeto (cubo o esfera)
def change_size(size):
    global current_size, scaled_cube_vertices, rotated_vertices, sphere_radius
    current_size = size
    if current_shape == 'cube':
        scaled_cube_vertices = scale_cube_vertices(size_options[current_size])
        rotated_vertices = apply_initial_rotation()
    else:
        sphere_radius = size_options[current_size]*2
        generate_sphere_ellipses()

# Listas para almacenar las líneas del usuario
user_lines = []

# Precisión inicial
accuracy = 0

# Variable para controlar si el botón de reset ha sido presionado
reset_button_pressed = False

reset_rect = None  # Inicializa fuera del bucle

# Bucle principal de la aplicación
running = True
drawing = False

cube_rect = None
sphere_rect = None

bandera_grid = False

while running:
    screen_size = screen.get_size()
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        elif event.type == VIDEORESIZE:
            if event.size != screen.get_size():
                try:
                    screen = crear_ventana(event.size)
                except pygame.error:
                    print("No se pudo recrear la ventana al redimensionar. Intentando de nuevo.")
                    screen = crear_ventana(screen.get_size())
            screen_size = screen.get_size()
     
            cache_dirty['cube'] = True
            cache_dirty['sphere'] = True
        elif event.type == MOUSEBUTTONDOWN:
            if event.button == 1:
                drawing = True
                x, y = event.pos
                screen_width, screen_height = screen.get_size()
                rect_width, rect_height = int(50 * screen_height / 600), int(90 * screen_height / 600)
                rotation_angle = 45  # Asegúrate de que coincida con el ángulo de rotación usado en draw_shape_buttons
                
                def point_in_polygon(point, polygon):
                    px, py = point
                    collision = False
                    for i, corner in enumerate(polygon):
                        next_corner = polygon[(i + 1) % len(polygon)]
                        if ((corner[1] > py) != (next_corner[1] > py)) and \
                                (px < (next_corner[0] - corner[0]) * (py - corner[1]) / (next_corner[1] - corner[1]) + corner[0]):
                            collision = not collision
                    return collision
       
                if point_in_polygon((x, y), cube_clickable):
                    current_shape = 'cube'
                    reset_application()
                
                elif point_in_polygon((x, y), sphere_clickable):
                    current_shape = 'sphere'
                    reset_application()
                    
                elif reset_rect and reset_rect.collidepoint(x, y):
                    reset_application()
                    
                elif drawing_mode == 'pencil':
                    user_lines.append({'points': [event.pos], 'brush_size': brush_size})
                elif drawing_mode == 'eraser':
                    pass
        elif event.type == MOUSEBUTTONUP:
            if event.button == 1:
                drawing = False
                if reset_button_pressed:
                    if reset_button.collidepoint(event.pos):
                        reset_application()
                    reset_button_pressed = False
                if user_lines and len(user_lines[-1]['points']) < 2:
                    user_lines.pop()
                #shape_points = generate_shape_points(screen_size)
                #accuracy = calculate_accuracy(user_lines, shape_points, screen_size)
        elif event.type == MOUSEMOTION and drawing:
            if drawing_mode == 'pencil' and user_lines:
                if user_lines:
                    user_lines[-1]['points'].append(event.pos)
                else:
                    user_lines.append({'points': [event.pos], 'brush_size': brush_size})
            elif drawing_mode == 'eraser':
                for line in user_lines:
                    for i in range(len(line['points']) - 1):
                        if np.linalg.norm(np.array(line['points'][i]) - np.array(event.pos)) < brush_size:
                            line['points'].clear()
                            break
        elif event.type == KEYDOWN:
            if event.key == K_SPACE:
                if not show_lines:
                    shape_points = generate_shape_points(screen_size)
                    accuracy = calculate_accuracy(user_lines, shape_points, screen_size)
                show_lines = not show_lines
                cache_dirty['cube'] = True
                cache_dirty['sphere'] = True
            elif event.key == K_r:
                reset_application()
            elif event.key == K_z and pygame.key.get_mods() & KMOD_CTRL:
                if user_lines:
                    user_lines.pop()
                cache_dirty['cube'] = True
                cache_dirty['sphere'] = True
            elif event.key == K_b:
                drawing_mode = 'pencil'
            elif event.key == K_e:
                drawing_mode = 'eraser'
            elif event.key == K_KP_PLUS or event.key == K_EQUALS:
                brush_size = min(5, brush_size + 1)
            elif event.key == K_KP_MINUS or event.key == K_MINUS:
                brush_size = max(1, brush_size - 1)
            elif event.key == K_1:
                change_difficulty('dificil')
            elif event.key == K_2:
                change_difficulty('normal')
            elif event.key == K_3:
                change_difficulty('facil')
            elif event.key == K_4:
                change_size('small')
            elif event.key == K_5:
                change_size('medium')
            elif event.key == K_6:
                change_size('large')
            elif event.key == K_t:
                plt.show()
            elif event.key == K_g:
                bandera_grid = not bandera_grid
            elif event.key == K_m:
                
                
                show_sphere_lines = not show_sphere_lines 
                cache_dirty['cube'] = True
                cache_dirty['sphere'] = True                             
    try:
        if screen:
            pygame.display.update()
    except pygame.error:
        print("Error al actualizar la pantalla. Intentando recrear la ventana.")
        screen = crear_ventana(screen_size)
    

    screen.fill(WHITE)
    #draw_grid(screen, camera_pos, screen_size, 256, 5)
    # Dibuja la forma y guarda en el caché si es necesario
    if current_shape == 'cube':
        if cube_cache_surface is None or cache_dirty['cube']:
            cube_cache_surface = pygame.Surface(screen_size, pygame.SRCALPHA)
            cube_cache_surface.fill((0, 0, 0, 0))  # Limpiar la superficie con transparencia
            draw_cube(show_lines, screen_size)  # Dibuja en la superficie principal para caché
            cube_cache_surface.blit(screen, (0, 0))  # Copiar a la superficie de caché
            cache_dirty['cube'] = False
        screen.blit(cube_cache_surface, (0, 0))
    else:
        if sphere_cache_surface is None or cache_dirty['sphere']:
            sphere_cache_surface = pygame.Surface(screen_size, pygame.SRCALPHA)
            sphere_cache_surface.fill((0, 0, 0, 0))  # Limpiar la superficie con transparencia

            
            draw_sphere(show_lines, screen_size)  # Dibuja en la superficie principal para caché


            sphere_cache_surface.blit(screen, (0, 0))  # Copiar a la superficie de caché
            cache_dirty['sphere'] = False
        screen.blit(sphere_cache_surface, (0, 0))
    if bandera_grid:
        draw_grid(screen, camera_pos, screen_size, 256, 5)



            
    draw_shape_buttons(screen)
    cube_rect, sphere_rect = draw_shape_buttons(screen)  # Actualizar los rectángulos de los botones
    reset_rect = draw_difficulty_size_reset(screen)
    draw_accuracy(screen)
    draw_graph(screen)
    cube_clickable, sphere_clickable = draw_shape_buttons(screen)
    # Dibujar el rombo en la esquina superior izquierda
    #draw_rhombus(screen, rhombus_size, rhombus_x, rhombus_y, rhombus_color)

    # Dibujar las líneas del usuario
    for line in user_lines:
        if len(line['points']) >= 2:
            for i in range(len(line['points']) - 1):
                pygame.draw.line(screen, BLACK, line['points'][i], line['points'][i + 1], line['brush_size'])
            pygame.draw.aalines(screen, BLACK, False, line['points'], 1)
        elif len(line['points']) == 1:
            pygame.draw.circle(screen, BLACK, line['points'][0], line['brush_size'])
            pygame.draw.circle(screen, WHITE, line['points'][0], line['brush_size'] - 1)


    # Botón de reinicio
    reset_button_width = int(screen_size[0] * 0.1)
    reset_button_height = int(screen_size[1] * 0.05)
    reset_button = pygame.Rect(screen_size[0] - reset_button_width - 10, 10, reset_button_width, reset_button_height)


    # Dibujar botones para cambiar entre cubo y esfera
    button_width = int(screen_size[0] * 0.1)
    button_height = int(screen_size[1] * 0.05)
    cube_button = pygame.Rect(screen_size[0] - button_width - 10, screen_size[1] - button_height * 2 - 20, button_width, button_height)
    sphere_button = pygame.Rect(screen_size[0] - button_width - 10, screen_size[1] - button_height - 10, button_width, button_height)
    
 

    # Dibujar el círculo indicador del tamaño del pincel en la posición del cursor
    mouse_pos = pygame.mouse.get_pos()
    if drawing_mode == 'pencil':
        pygame.draw.circle(screen, BLACK, mouse_pos, brush_size)
    elif drawing_mode == 'eraser':
        pygame.draw.circle(screen, WHITE, mouse_pos, brush_size)
        pygame.draw.circle(screen, BLACK, mouse_pos, brush_size, 1)

    pygame.display.update()
    clock.tick(60)

pygame.quit()
