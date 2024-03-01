
from PIL import Image, ImageDraw
import myCanny
import numpy as np
from matplotlib import pyplot as plt
import imageio



def hough_transform(img_bin, theta_res=1, rho_res=1):
  
  M,N = img_bin.shape
  theta = np.linspace(-90.0, 0.0, np.ceil(90.0/theta_res) + 1.0)
  theta = np.concatenate((theta, -theta[len(theta)-2::-1]))
  Diagonal = np.sqrt((M - 1)**2 + (N - 1)**2)
  q = np.ceil(Diagonal/rho_res)
  nrho = 2*q + 1
  rho = np.linspace(-q*rho_res, q*rho_res, nrho)
  H = np.zeros((len(rho), len(theta)))
  for rowIdx in range(M):
    for colIdx in range(N):
      if img_bin[rowIdx, colIdx]:
        for thIdx in range(len(theta)):
          rhoVal = colIdx*np.cos(theta[thIdx]*np.pi/180.0) + rowIdx*np.sin(theta[thIdx]*np.pi/180)
          rhoIdx = np.nonzero(np.abs(rho-rhoVal) == np.min(np.abs(rho-rhoVal)))[0]
          H[rhoIdx[0], thIdx] += 1
  return rho, theta, H

def top_n_rho_theta_pairs(ht_acc_matrix, n, rhos, thetas):
  
  flat = list(set(np.hstack(ht_acc_matrix)))
  flat_sorted = sorted(flat, key = lambda n: -n)
  coords_sorted = [(np.argwhere(ht_acc_matrix == acc_value)) for acc_value in flat_sorted[0:n]]
  rho_theta = []
  x_y = []
  for coords_for_val_idx in range(0, len(coords_sorted), 1):
    coords_for_val = coords_sorted[coords_for_val_idx]
    for i in range(0, len(coords_for_val), 1):
      n,m = coords_for_val[i] # n by m matrix
      rho = rhos[n]
      theta = thetas[m]
      rho_theta.append([rho, theta])
      x_y.append([m, n]) # just to unnest and reorder coords_sorted
  return [rho_theta[0:n], x_y]

def valid_point(pt, ymax, xmax):
  '''
  @return True/False if pt is with bounds for an xmax by ymax image
  '''
  x, y = pt
  if x <= xmax and x >= 0 and y <= ymax and y >= 0:
    return True
  else:
    return False

def round_tup(tup):
  '''
  @return closest integer for each number in a point for referencing
  a particular pixel in an image
  '''
  x,y = [int(round(num)) for num in tup]
  return (x,y)

def draw_rho_theta_pairs(target_im, pairs):
  '''
  @param opencv image
  @param array of rho and theta pairs
  Has the side-effect of drawing a line corresponding to a rho theta
  pair on the image provided
  '''
  im_y_max, im_x_max, channels = np.shape(target_im)
  img = Image.fromarray(target_im) 
  output_image = Image.new("RGB", img.size)
  output_image.paste(img)
  draw_result = ImageDraw.Draw(output_image)

  for i in range(0, len(pairs), 1):
    point = pairs[i]
    rho = point[0]
    theta = point[1] * np.pi / 180 # degrees to radians
    # y = mx + b form
    m = -np.cos(theta) / np.sin(theta)
    b = rho / np.sin(theta)
    # possible intersections on image edges
    left = (0, b)
    right = (im_x_max, im_x_max * m + b)
    top = (-b / m, 0)
    bottom = ((im_y_max - b) / m, im_y_max)

    pts = [pt for pt in [left, right, top, bottom] if valid_point(pt, im_y_max, im_x_max)]
    if len(pts) == 2:
      x1,y1=round_tup(pts[0])
      x2,y2=round_tup(pts[1])
      draw_result.line( (x1,y1,x2,y2), fill= 128)

  return output_image
      
def hough_line(img,edges):

  rhos, thetas, H = hough_transform(edges)
  rho_theta_pairs, x_y_pairs = top_n_rho_theta_pairs(H, 22, rhos, thetas)
  im_w_lines = img.copy()
  output_image = draw_rho_theta_pairs(im_w_lines, rho_theta_pairs)
  return np.array(output_image),H
