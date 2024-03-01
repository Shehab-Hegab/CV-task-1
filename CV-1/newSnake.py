import numpy as np
import scipy.linalg
import scipy.ndimage
import skimage
import skimage.filters
import scipy.interpolate


def kassSnake(image, initialContour, edgeImage=None, alpha=0.01, beta=0.1, wLine=0, wEdge=1, gamma=0.01,
              maxPixelMove=None, maxIterations=2500, convergence=0.1):
    maxIterations = int(maxIterations)
    if maxIterations <= 0:
        raise ValueError('maxIterations should be greater than 0.')

    convergenceOrder = 10

    # valid_bcs = ['periodic', 'free', 'fixed', 'free-fixed',
    #              'fixed-free', 'fixed-fixed', 'free-free']
    # if bc not in valid_bcs:
    #     raise ValueError("Invalid boundary condition.\n" +
    #                      "Should be one of: " + ", ".join(valid_bcs) + '.')

    image = skimage.img_as_float(image)
    isMultiChannel = image.ndim == 3

    # If edge image is not given and an edge weight is specified, then get the edge of image using sobel mask
    # Otherwise set edge image to zero if it is none (it follows that wEdge must be 0)
    if edgeImage is None and wEdge != 0:
        # Reflect mode is used to minimize the values at the outer boundaries of the edge image
        # When applying a Sobel kernel, there are a few ways to handle border, reflect repeats the outside
        # edges which should return a small edge
        edgeImage = np.sqrt(scipy.ndimage.sobel(image, axis=0, mode='reflect') ** 2 +
                            scipy.ndimage.sobel(image, axis=1, mode='reflect') ** 2)

        # Normalize the edge image between [0, 1]
        edgeImage = (edgeImage - edgeImage.min()) / (edgeImage.max() - edgeImage.min())
    elif edgeImage is None:
        edgeImage = 0

    # Calculate the external energy which is composed of the image intensity and ege intensity
    # TODO Add termination energy
    # TODO Add constraint energy
    if isMultiChannel:
        externalEnergy = wLine * np.sum(image, axis=2) + wEdge * np.sum(edgeImage, axis=2)
    else:
        externalEnergy = wLine * image + wEdge * edgeImage

    # Take external energy array and perform interpolation over the 2D grid
    # If a fractional x or y is requested, then it will interpolate between the intensity values surrounding the point
    # This is an object that can be given an array of points repeatedly
    externalEnergyInterpolation = scipy.interpolate.RectBivariateSpline(np.arange(externalEnergy.shape[1]),
                                                                        np.arange(externalEnergy.shape[0]),
                                                                        externalEnergy.T, kx=2, ky=2, s=0)

    # Split initial contour into x's and y's
    x, y = initialContour[:, 0].astype(float), initialContour[:, 1].astype(float)

    # Create a matrix that will contain previous x/y values of the contour
    # Used to determine if contour has converged if the previous values are consistently smaller
    # than the convergence amount
    previousX = np.empty((convergenceOrder, len(x)))
    previousY = np.empty((convergenceOrder, len(y)))

    # Build snake shape matrix for Euler equation
    # This matrix is used to calculate the internal energy in the snake
    # This matrix can be obtained from Equation 14 in Appendix A from Kass paper (1988)
    # r is the v_{i} components grouped together
    # q is the v_{i-1} components grouped together (and v_{i+1} components are the same)
    # p is the v_{i-2} components grouped together (and v_{i+2} components are the same)
    n = len(x)
    r = 2 * alpha + 6 * beta
    q = -alpha - 4 * beta
    p = beta

    A = r * np.eye(n) + \
        q * (np.roll(np.eye(n), -1, axis=0) + np.roll(np.eye(n), -1, axis=1)) + \
        p * (np.roll(np.eye(n), -2, axis=0) + np.roll(np.eye(n), -2, axis=1))

    # TODO Impose boundary conditions, fixed or free instead of periodic
    # # Impose boundary conditions different from periodic:
    # sfixed = False
    # if bc.startswith('fixed'):
    #     A[0, :] = 0
    #     A[1, :] = 0
    #     A[1, :3] = [1, -2, 1]
    #     sfixed = True
    # efixed = False
    # if bc.endswith('fixed'):
    #     A[-1, :] = 0
    #     A[-2, :] = 0
    #     A[-2, -3:] = [1, -2, 1]
    #     efixed = True
    # sfree = False
    # if bc.startswith('free'):
    #     A[0, :] = 0
    #     A[0, :3] = [1, -2, 1]
    #     A[1, :] = 0
    #     A[1, :4] = [-1, 3, -3, 1]
    #     sfree = True
    # efree = False
    # if bc.endswith('free'):
    #     A[-1, :] = 0
    #     A[-1, -3:] = [1, -2, 1]
    #     A[-2, :] = 0
    #     A[-2, -4:] = [-1, 3, -3, 1]
    #     efree = True

    # TODO Cite papers better here

    # Invert matrix once since alpha, beta and gamma are constants
    # See equation 19 and 20 in Appendix A of Kass's paper
    AInv = scipy.linalg.inv(A + gamma * np.eye(n))
    # AInv = scipy.linalg.inv(gamma * A + np.eye(n))

    for i in range(maxIterations):
        # Calculate the gradient in the x/y direction of the external energy
        fx = externalEnergyInterpolation(x, y, dx=1, grid=False)
        fy = externalEnergyInterpolation(x, y, dy=1, grid=False)

        # TODO Figure out the purpose of this
        # if sfixed:
        #     fx[0] = 0
        #     fy[0] = 0
        # if efixed:
        #     fx[-1] = 0
        #     fy[-1] = 0
        # if sfree:
        #     fx[0] *= 2
        #     fy[0] *= 2
        # if efree:
        #     fx[-1] *= 2
        #     fy[-1] *= 2

        # Compute new x and y contour
        # See Equation 19 and 20 in Appendix A of Kass's paper
        xNew = np.dot(AInv, gamma * x + fx)
        # xNew = np.dot(AInv, x + gamma * fx)
        yNew = np.dot(AInv, gamma * y + fy)
        # yNew = np.dot(AInv, y + gamma * fy)

        # Maximum pixel move sets a cap on the maximum amount of pixels that one step can take.
        # This is useful if one needs to prevent the snake from jumping past the location minimum one desires.
        # In many cases, it is better to leave it off to increase the speed of the algorithm

        # Calculated by getting the x and y delta from the new points to previous points
        # Then get the angle of change and apply maxPixelMove magnitude
        # Otherwise, if no maximum pixel move is set then set the x/y to be xNew/yNew
        if maxPixelMove:
            # print('test')
            dx = maxPixelMove * np.tanh(xNew - x)
            dy = maxPixelMove * np.tanh(yNew - y)

            x += dx
            y += dy
        else:
            x = xNew
            y = yNew

        # TODO Figure this out
        # if sfixed:
        #     dx[0] = 0
        #     dy[0] = 0
        # if efixed:
        #     dx[-1] = 0
        #     dy[-1] = 0

        # j is variable that loops around from 0 to the convergence order. This is used to save the previous value
        # Convergence is reached when absolute value distance between previous values and current values is less
        # than convergence threshold
        # Note: Max on axis 1 and then min on the 0 axis for distance. Retrieves maximum distance from the contour
        # for each trial, and then gets the minimum of the 10 trials.
        j = i % (convergenceOrder + 1)

        if j < convergenceOrder:
            previousX[j, :] = x
            previousY[j, :] = y
        else:
            distance = np.min(np.max(np.abs(previousX - x[None, :]) + np.abs(previousY - y[None, :]), axis=1))

            if distance < convergence:
                break

    print('Finished at', i)

    return np.array([x, y]).T
