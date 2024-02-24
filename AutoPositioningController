package frc.robot.vision;

import java.util.List;

import edu.wpi.first.math.controller.PIDController;
import edu.wpi.first.math.controller.RamseteController;
import edu.wpi.first.math.controller.SimpleMotorFeedforward;
import edu.wpi.first.math.geometry.Pose2d;
import edu.wpi.first.math.geometry.Rotation2d;
import edu.wpi.first.math.geometry.Translation2d;
import edu.wpi.first.math.trajectory.Trajectory;
import edu.wpi.first.math.trajectory.TrajectoryConfig;
import edu.wpi.first.math.trajectory.TrajectoryGenerator;
import edu.wpi.first.math.trajectory.constraint.DifferentialDriveVoltageConstraint;
import edu.wpi.first.wpilibj2.command.CommandBase;
import edu.wpi.first.wpilibj2.command.RamseteCommand;
import edu.wpi.first.wpilibj2.command.SequentialCommandGroup;
import frc.robot.Constants.MotionControl;
import frc.robot.subsystems.DrivetrainSubsystem;
import frc.robot.Constants.*;

public class AutoPositioningController extends CommandBase{
    private final DrivetrainSubsystem m_drivetrain;
    private final VisionController m_vision;

    public AutoPositioningController(DrivetrainSubsystem drivetrain, VisionController vision){
        m_drivetrain = drivetrain;
        m_vision = vision;
    }

    public DrivetrainSubsystem getDrivetrain (){
        return(m_drivetrain);
    }

    public VisionController getVision (){
        return(m_vision);
    }
  
    public SequentialCommandGroup getAutonomousCommand() {
        // Create voltage constraint to ensure we don't accelerate too fast
        var autoVoltageConstraint = 
            new DifferentialDriveVoltageConstraint(
                new SimpleMotorFeedforward(
                    MotionControl.ksVolts,
                    MotionControl.kvVoltSecondsPerMeter,
                    MotionControl.kaVoltSecondsSquaredPerMeter),
                MotionControl.kDriveKinematics,
                SubsystemConfig.AUTO_MAX_VOLTAGE);

        // Create config for trajectory
        TrajectoryConfig config = 
            new TrajectoryConfig(
                    MotionControl.kMaxSpeedMetersPerSecond,
                    MotionControl.kMaxAccelerationMetersPerSecondSquared)
            // Add kinematics to ensure max speed is actually obeyed
            .setKinematics(MotionControl.kDriveKinematics)
            // Apply the voltage constraint
            .addConstraint(autoVoltageConstraint);

        //Arjun does not like to have long lines of code so instead here are the variables that I created for him to shorten the lines of code to prevent long lines of code from being created because as mentioned Arjun dislikes long lines of code. To be completely clear Arjun Parthasarathy does not like long lines of code and thus the code below was made to prevent this.
        double startX = m_drivetrain.getPose().getX();
        double startY = m_drivetrain.getPose().getY();
        
        double targetX = m_vision.getTranslationToTarget().getX();
        double targetY = m_vision.getTranslationToTarget().getY();

        double multiple = 2/3;

        double x1 = startX + multiple*(targetX-startX); 
        double y1 = startY + multiple*(targetY-startY);

        // Placeholder positions
        Trajectory visionTrajectory = 
            TrajectoryGenerator.generateTrajectory(
                //The starting position as taken from drivetrain
                m_drivetrain.getPose(),
                //The middle point as calculated above
                List.of(new Translation2d(x1, y1)),
                //The final position as calculated above
                new Pose2d(targetX, targetY, new Rotation2d(0)),
                //Smash or Pass config
                config);
        
        RamseteCommand ramseteCommand =
            new RamseteCommand(
                visionTrajectory,
                m_drivetrain::getPose,
                new RamseteController(),
                new SimpleMotorFeedforward(
                    MotionControl.ksVolts,
                    MotionControl.kvVoltSecondsPerMeter),
                MotionControl.kDriveKinematics,
                m_drivetrain::getWheelSpeeds,
                new PIDController(MotionControl.drivekP, 0, MotionControl.drivekD),
                new PIDController(MotionControl.drivekP, 0, MotionControl.drivekD),
                // RamseteCommand passes volts to the callback
                m_drivetrain::tankDriveVolts,
                m_drivetrain);
                
        // Reset odometry to the starting pose of the trajectory
        m_drivetrain.resetOdometry(visionTrajectory.getInitialPose());

        // Run path following command, then stop at the end
        return ramseteCommand.andThen(() -> m_drivetrain.tankDriveVolts(0, 0));
    }
}
