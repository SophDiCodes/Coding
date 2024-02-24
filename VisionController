package frc.robot.vision;

import org.photonvision.PhotonCamera;
import org.photonvision.PhotonUtils;
import org.photonvision.targeting.PhotonPipelineResult;
import org.photonvision.targeting.PhotonTrackedTarget;

import edu.wpi.first.math.geometry.Rotation2d;
import edu.wpi.first.math.geometry.Translation2d;

import static frc.robot.Constants.*;

public class VisionController {

    PhotonCamera m_camera;

    public VisionController() {
        m_camera = new PhotonCamera("OV5647");
    }
    
    public PhotonPipelineResult getLatestResult() {
        return m_camera.getLatestResult();
    }

    public boolean hasTargets(PhotonPipelineResult result) {
        return result.hasTargets();
    }

    public PhotonTrackedTarget getTarget(PhotonPipelineResult result) {
        return result.getBestTarget();
    }

    // Calculates how far in front of the robot the target is
    public double getDistanceToTarget() {
        PhotonPipelineResult result = this.getLatestResult();
        if (this.hasTargets(result)) {
            double range =
                PhotonUtils.calculateDistanceToTargetMeters(
                        RobotMeasurements.CAMERA_HEIGHT_METERS,
                        RobotMeasurements.TARGET_HEIGHT_METERS,
                        RobotMeasurements.CAMERA_PITCH_RADIANS,
                        Units.DEGREE.to(Units.RADIAN) * result.getBestTarget().getPitch());
            
            return range;
        }
        else {
            // No target found in frame
            return -1.0;
        }
    }

    public Translation2d getTranslationToTarget() {
        PhotonPipelineResult result = this.getLatestResult();
        if (this.hasTargets(result)) {
            double yDistanceToTarget = this.getDistanceToTarget();
            Translation2d translation = PhotonUtils.estimateCameraToTargetTranslation(
            yDistanceToTarget, Rotation2d.fromDegrees(-result.getBestTarget().getYaw()));

            return translation;
        }
        else {
            // If we can't find target in frame
            return new Translation2d();
        }
        //TODO figure out if we're using a RamseteCommand to get to the target or just PID
    }

    public double getHorizontalAngle() {
        PhotonPipelineResult result = this.getLatestResult();
        if (this.hasTargets(result)) {
            return (result.getBestTarget().getYaw());
        } else {
            return -1.0;
        }

    }

    public double getVerticalAngle() {
        PhotonPipelineResult result = this.getLatestResult();
        if (this.hasTargets(result)) {
            return (result.getBestTarget().getPitch());
        } else {
            return -1.0;
        }

    }

}
