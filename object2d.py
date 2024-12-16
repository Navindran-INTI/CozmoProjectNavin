import math
import numpy

from frame2d import Frame2D
from cozmo.util import Pose, Position

# function to determine intersecting polygonal objects. pos1 and pos2 are 2 objects defined by a
# matrix of [x,y,theta] columns and a number of rows equal to the number of corners of the polygon.
# Straight lines connect the corners, which are assumed to be in order, thus pos[:,1] connects to
# pos[:,2]. The last corner connects with the first to form a closed shape. Both poses must be in
# the same coordinate reference frame.
def ObjectsIntersect(pos1,pos2):
    # set up to get the slopes for 2-point form linear equations
    secondPoints1 = numpy.roll(pos1,-1,1)
    secondPoints2 = numpy.roll(pos2,-1,1)
    positions2 = numpy.stack([numpy.roll(pos2,-places,1) for places in pos2.shape[1]])
    diffs1 = secondPoints1-pos1
    diffs2 = positions2[1,:,:]-pos2
    slopes1 = diffs1[1,:]/diffs1[0,:]
    slopes2 = diffs2[1,:]/diffs2[0,:]
    permutedSlopes2 = numpy.stack([numpy.roll(slopes2,-places,1) for places in slopes2.shape[0]])
    # now, solve for the intersection of the extended lines from each side of the polygon
    # for all possible permutations of which line intersects which. We rely heavily on
    # numpy broadcasting to do the grunt work here. At the end what we should end up with
    # is a num_sides*num_sides matrix of possible intercepts
    interceptsx = positions2[:,1,:]-pos1[1,:]+slopes1*pos1[0,:]-permutedSlopes2*pos2[:,0,:]
    interceptsx = interceptsx/(slopes1-permutedSlopes2)
    interceptsy = slopes1*(interceptsx-pos1[0,:])+pos1[1,:]
    # the objects will intersect if there is a joint point which lies between the corners
    # for any side of the polygon. So check whether the intersection is between corners
    # for each solved system above. First step, find out how the intercept relates to
    # each corner
    conditionsx1A = secondPoints1[0,:] >= interceptsx
    conditionsx1B = interceptsx >= pos1[0,:]
    conditionsx1C = secondPoints1[0,:] <= interceptsx
    conditionsx1D = interceptsx <= pos1[0,:]
    conditionsx2A = numpy.roll(positions2[:,0,:],-1,0) >= interceptsx
    conditionsx2B = interceptsx >= positions2[:,0,:]
    conditionsx2C = numpy.roll(positions2[:,0,:],-1,0) <= interceptsx
    conditionsx2D = interceptsx <= positions2[:,0,:]
    conditionsy1A = secondPoints1[1,:] >= interceptsy
    conditionsy1B = interceptsy >= pos1[0,:]
    conditionsy1C = secondPoints1[1,:] <= interceptsy
    conditionsy1D = interceptsy <= pos1[0,:]
    conditionsy2A = numpy.roll(positions2[:,1,:],-1,0) >= interceptsy
    conditionsy2B = interceptsy >= positions2[:,0,:]
    conditionsy2C = numpy.roll(positions2[:,1,:],-1,0) <= interceptsy
    conditionsy2D = interceptsy <= positions2[:,0,:]
    # now, test whether it is between corners
    betweenx1A = conditionsx1A and conditionsx1B
    betweenx1B = conditionsx1C and conditionsx1D
    betweenx2A = conditionsx2A and conditionsx2B
    betweenx2B = conditionsx2C and conditionsx2D
    betweeny1A = conditionsy1A and conditionsy1B
    betweeny1B = conditionsy1C and conditionsy1D
    betweeny2A = conditionsy2A and conditionsy2B
    betweeny2B = conditionsy2C and conditionsy2D
    # next, merge the possible conditions for each polygon.
    betweenx2 = betweenx2A or betweenx2B
    betweeny2 = betweeny2A or betweeny2B
    intersectionsx = (betweenx1A or betweenx1B) and betweenx2
    intersectionsy = (betweeny1A or betweeny1B) and betweeny2
    # and finally if both X and Y match for some segment combination, we have
    # an intersection.
    return numpy.any(intersectionsx and intersectionsy)

# an Object2D is a representation of a space-occupying object in the Cozmo world.
# its pose is a Frame2D, and it has x and y sizes which give it an implied bounding
# box. Most objects are going to be rectangular, so the bounding box can be used to
# get a reasonable approximation of the space occupied by the object, or in a different
# context, how much room the object occupies in the Cozmo FOV.
class Object2D:
      def __init__(self, pose: Frame2D, xdim, ydim):
          self.pose = pose
          self.xdim = xdim
          self.ydim = ydim

          self.xform = self.pose.inverse()
          # matrix representation as a block of coordinates
          # self.mat = numpy.matmul(self.pose.mat,numpy.array([[-self.xdim/2,-self.ydim/2,1],[self.xdim/2,-self.ydim/2,1],[self.xdim/2,self.ydim/2,1],[-self.xdim/2,self.ydim/2,1]]))
          self._diag = math.sqrt(self.xdim**2+self.ydim**2)

      def front(self):
          frontIdio = Frame2D.fromXYA(self.xdim/2,0,self.pose.angle())
          return self.pose.mult(frontIdio)
      
      def back(self):
          backIdio = Frame2D.fromXYA(-self.xdim/2,0,self.pose.angle())
          return self.pose.mult(backidio)

      def left(self):
          leftIdio = Frame2D.fromXYA(0,-self.ydim/2,self.pose.angle())
          return self.pose.mult(leftIdio)

      def right(self):
          rightIdio = Frame2D.fromXYA(0,self.ydim/2,self.pose.angle())
          return self.pose.mult(rightIdio)

      # object's pose, as seen in the reference frame of another object
      def poseRelative(self,other: Frame2D):
          return other.inverse().mult(self.pose)
          
      def frontRelative(self,other: Frame2D):
          # first, get our pose in the reference frame of the other object
          relativePose = self.poseRelative(other)
          ''' Angles are positive in the anticlockwise
              direction, negative in the clockwise direction, and are in
              the range [-pi/2,pi/2]. 
              We simply look for what will be the most
              forward coordinate in each rotational region, generally, a corner or
              the centre of a face. Note that we are looking for front, back, left
              and right relative to the other object - that is, as if the other
              object was facing this one. In this reference, back is front and front
              is back, under no rotation
          '''
          if abs(relativePose.angle) == math.pi:
             return otherInv.mult(self.back())
          elif relativePose.angle > math.pi/2:
             frontIdio = Frame2D.fromXYA(self.xdim/2,-self.ydim/2)
             return otherInv.mult(frontIdio)
          elif relativePose.angle == math.pi/2:
             return otherInv.mult(self.left())
          elif relativePose.angle > 0:
             frontIdio = Frame2D.fromXYA(-self.xdim/2,-self.ydim/2)
             return otherInv.mult(frontIdio)
          elif relativePose.angle == 0:
             return otherInv.mult(self.back())
          elif relativePose.angle > -math.pi/2:
             frontIdio = Frame2D.fromXYA(-self.xdim/2,self.ydim/2)
          elif relativePose.angle == -math.pi/2:
             return otherinv.mult(self.right())
          else:
             frontIdio = Frame2D.fromXYA(self.xdim/2,self.ydim/2)
             return otherinv.mult(frontIdio)

      def backRelative(self,other: Frame2D):
          otherInv = other.inverse()
          relativePose = otherInv.mult(self.pose)
          # uses all the same conventions as FrontRelative, the only difference being
          # that we are now trying to find the most rearward coordinate.
          if abs(relativePose.angle) == math.pi:
             return otherInv.mult(self.back())
          elif relativePose.angle > math.pi/2:
             backIdio = Frame2D.fromXYA(-self.xdim/2,self.ydim/2)
             return otherInv.mult(backIdio)
          elif relativePose.angle == math.pi/2:
             return otherInv.mult(self.right())
          elif relativePose.angle >  0:
             backIdio = Frame2D.fromXYA(self.xdim/2,self.ydim/2)
             return otherInv.mult(backIdio)
          elif relativePose.angle == 0:
             return otherInv.mult(self.front())
          elif relativePose.angle > -math.pi/2:
             backIdio = Frame2D.fromXYA(self.xdim/2,-self.ydim/2)
          elif relativePose.angle == -math.pi/2:
             return otherinv.mult(self.left())
          else:
             backIdio = Frame2D.fromXYA(-self.xdim/2,-self.ydim/2)
             return otherinv.mult(backIdio)

      def leftRelative(self,other: Frame2D):
          otherInv = other.inverse()
          relativePose = otherInv.mult(self.pose)
          # same thing here, now for the left coordinate.
          if abs(relativePose.angle) == math.pi:
             return otherInv.mult(self.right())
          elif relativePose.angle > math.pi/2:
             backIdio = Frame2D.fromXYA(self.xdim/2,self.ydim/2)
             return otherInv.mult(backIdio)
          elif relativePose.angle == math.pi/2:
             return otherInv.mult(self.front())
          elif relativePose.angle > 0:
             backIdio = Frame2D.fromXYA(self.xdim/2,-self.ydim/2)
             return otherInv.mult(backIdio)
          elif relativePose.angle == 0:
             return otherInv.mult(self.left())
          elif relativePose.angle > -math.pi/2:
             backIdio = Frame2D.fromXYA(-self.xdim/2,-self.ydim/2)
          elif relativePose.angle == -math.pi/2:
             return otherinv.mult(self.back())
          else:
             backIdio = Frame2D.fromXYA(-self.xdim/2,self.ydim/2)
             return otherinv.mult(backIdio)

      def rightRelative(self,other: Frame2D):
          otherInv = other.inverse()
          relativePose = otherInv.mult(self.pose)
          # and again, for the right coordinate.
          if abs(relativePose.angle) == math.pi:
             return otherInv.mult(self.left())
          elif relativePose.angle > math.pi/2:
             backIdio = Frame2D.fromXYA(-self.xdim/2,-self.ydim/2)
             return otherInv.mult(backIdio)
          elif relativePose.angle == math.pi/2:
             return otherInv.mult(self.back())
          elif relativePose.angle > 0:
             backIdio = Frame2D.fromXYA(-self.xdim/2,self.ydim/2)
             return otherInv.mult(backIdio)
          elif relativePose.angle == 0:
             return otherInv.mult(self.right())
          elif relativePose.angle > -math.pi/2:
             backIdio = Frame2D.fromXYA(self.xdim/2,self.ydim/2)
          elif relativePose.angle == -math.pi/2:
             return otherinv.mult(self.front())
          else:
             backIdio = Frame2D.fromXYA(self.xdim/2,-self.ydim/2)
             return otherinv.mult(backIdio)

      # 'line-of-sight' front, left, and right indicate which extremum
      # would be visible in any line of sight drawn from the robot (whether
      # that line actually is within the visual field or not)
      def frontLOS(self,robot: Frame2D):
          maxFront = self.frontRelative(robot)
          maxLeft = self.leftRelative(robot)
          maxRight = self.rightRelative(robot)
          angleFront = np.arctan2(maxFront.y(),maxFront.x())
          angleLeft = np.arctan2(maxLeft.y(),maxLeft.x())
          angleRight = np.arctan2(maxRight.y(),maxRight.x())
          if angleFront < 0:
             if angleLeft < angleFront:
                return maxLeft
          if angleFront > 0:
             if angleRight > angleFront:
                return maxRight
          return maxFront
                      
      def leftLOS(self,robot: Frame2D):
          frontTgt = self.frontRelative(robot)
          leftTgt = self.leftRelative(robot)
          angleFront = np.arctan2(frontTgt.y(),frontTgt.x())
          angleLeft = np.arctan2(leftTgt.y(),leftTgt.x())
          if (angleFront > 0) and (angleFront > angleLeft):
             return frontTgt
          return leftTgt
          
      def rightLOS(self,robot: Frame2D):
          frontTgt = self.frontRelative(robot)
          rightTgt = self.rightRelative(robot)
          angleFront = np.arctan2(frontTgt.y(),frontTgt.x())
          angleRight = np.arctan2(rightTgt.y(),rightTgt.x())
          if (angleFront < 0) and (angleFront < angleRight):
             return frontTgt
          return rightTgt

      # is any part of the object visible? The trick here is to notice that the
      # only way it can be completely invisible is if both left and right line-of-sight
      # points are on the same side of the visual field and out of the field of view
      def objectVisible(self,robot: Frame2D):
          left = self.leftLOS(robot)
          right = self.rightLOS(robot)
          angleLeft = np.arctan2(left.y(),left.x())
          angleRight = np.arctan2(right.y(),right.x())
          # anything behind the robot itself is clearly not visible
          if left.x() < 0 and right.x() < 0:
             return False
          # the copysign function below in the crucial test for visibility implements
          # sgn(x) == sgn(y). This odd expression is needed because the core Python devs
          # have for whatever reason chosen not to implement an sgn() function (and a
          # brief search uncovers evidence of an ideological battle at some point in the
          # past)
          if (abs(angleLeft) > 1/6*math.pi and
              abs(angleRight) > 1/6*math.pi and
              math.copysign(1,angleLeft) == math.copysign(1,angleRight)):
             return False
          return True

      # determine the distance to the object (target), relative to an arbitrary angle from some
      # other object. This is NOT the same as the shortest distance between these objects
      def distanceAtAngle(self,other,angle):
          # first step: get the left and right LOS points to get a target face
          left = self.leftLOS(other)
          right = self.rightLOS(other)
          # first step: extract the target's intercept point with the on-axis LOS of the object
          tgtFaceSlope = (right.y()-left.y())/(right.x()-left.x())
          xOnAxis = left.x()*tgtFaceSlope-left.y()/tgtFaceSlope
          # second step: get the half of the inner triangle from the object to the intercept
          delta = xOnAxis*math.cos(angle)
          # final step: get the remaining part of the distance line, which is the other half
          # of the inner triangle. We can do this because we know the relative pose of the
          # target object, which gives us the 'third angle' of the inner triangle
          gamma = xOnAxis*math.sin(angle)
          innerAngle = math.pi()-angle-tgtObject[1].angle()
          epsilon = gamma/math.tan(innerAngle)
          return delta+epsilon

      # get which is the leftmost visible point of this object, within the robot's reference frame
      def leftVisible(self,robot: Frame2D):
          maxLeft = self.leftLOS(robot)
          maxRight = self.rightLOS(robot)
          angleMaxLeft = np.arctan2(maxLeft.y(),maxLeft.x())
          angleMaxRight = np.arctan2(maxRight.y(),maxRight.x())
          if (abs(angleLeft) > 1/6*math.pi and
              abs(angleRight) > 1/6*math.pi and
              math.copysign(1,angleLeft) == math.copysign(1,angleRight)):
             return None
          if abs(angleMaxLeft) <= 1/6*math.pi:
             return maxLeft
          distanceRoboLeft = self.distanceAtAngle(robot,1/6*math.pi)
          return Frame2D(distanceRoboLeft*math.cos(1/6*math.pi),
                         distanceRoboLeft*math.sin(1/6*math.pi),
                         robot.inverse().mult(self.pose).angle())

      def rightVisible(self,robot: Frame2D):
          maxLeft = self.leftLOS(robot)
          maxRight = self.rightLOS(robot)
          angleMaxLeft = np.arctan2(maxLeft.y(),maxLeft.x())
          angleMaxRight = np.arctan2(maxRight.y(),maxRight.x())
          if (abs(angleLeft) > 1/6*math.pi and
              abs(angleRight) > 1/6*math.pi and
              math.copysign(1,angleLeft) == math.copysign(1,angleRight)):
             return None
          if abs(angleMaxRight) <= 1/6*math.pi:
             return maxRight
          distanceRoboRight = self.distanceAtAngle(robot,-1/6*math.pi)
          return Frame2D(distanceRoboRight*math.cos(-1/6*math.pi),
                         distanceRoboRight*math.sin(-1/6*math.pi),
                         robot.inverse().mult(self.pose).angle())
