package frc.robot.subsystems;

import com.ctre.phoenix.motorcontrol.ControlMode;
import com.ctre.phoenix.motorcontrol.DemandType;
import com.ctre.phoenix.motorcontrol.NeutralMode;
import com.ctre.phoenix.motorcontrol.can.TalonSRXConfiguration;
import com.ctre.phoenix.motorcontrol.can.WPI_TalonSRX;
import com.ctre.phoenix.motorcontrol.can.WPI_VictorSPX;
import com.kauailabs.navx.frc.AHRS;

import edu.wpi.first.wpilibj2.command.SubsystemBase;

import edu.wpi.first.math.kinematics.DifferentialDriveOdometry;
import edu.wpi.first.math.kinematics.DifferentialDriveWheelSpeeds;
import edu.wpi.first.wpilibj.smartdashboard.Field2d;
import edu.wpi.first.wpilibj.smartdashboard.SmartDashboard;
import edu.wpi.first.math.geometry.Pose2d;
import edu.wpi.first.math.geometry.Rotation2d;

import static frc.robot.Constants.*;

public class DrivetrainSubsystem extends SubsystemBase {

    private final WPI_VictorSPX m_leftFrontTalon, m_rightFrontTalon;
    private final WPI_TalonSRX m_leftRearTalon, m_rightRearTalon;
    private final AHRS m_gyro;
    private final DifferentialDriveOdometry m_odometry;
    private Field2d m_field = new Field2d();

    public DrivetrainSubsystem()
    {
        m_leftFrontTalon = new WPI_VictorSPX(Ports.CAN_DRIVETRAIN_LEFT_FRONT_VICTOR);
        m_leftRearTalon = new WPI_TalonSRX(Ports.CAN_DRIVETRAIN_LEFT_REAR_TALONSRX);
        m_rightFrontTalon = new WPI_VictorSPX(Ports.CAN_DRIVETRAIN_RIGHT_FRONT_VICTOR);
        m_rightRearTalon = new WPI_TalonSRX(Ports.CAN_DRIVETRAIN_RIGHT_REAR_TALONSRX);
        
        m_gyro = new AHRS(Ports.SPI_PORT_GYRO);
        m_gyro.reset();
        m_odometry = new DifferentialDriveOdometry(m_gyro.getRotation2d(), 0, 0, new Pose2d(0, 0, m_gyro.getRotation2d()));

        SmartDashboard.putData("Field", m_field);
        configureMotors();

        resetEncoders();
    }

    @Override
    public void periodic()
    {
        var gyroAngle = Rotation2d.fromDegrees(-m_gyro.getAngle());
        m_odometry.update(gyroAngle, getDistanceLeft(), getDistanceRight());
        m_field.setRobotPose(m_odometry.getPoseMeters());
    }

    public void drivePO(double left, double right)
    {
        m_leftRearTalon.set(ControlMode.PercentOutput, left);
        m_rightRearTalon.set(ControlMode.PercentOutput, right);
    }

    public void drive(double left, double right)
    {
        m_leftRearTalon.selectProfileSlot(1, MotorConfig.TALON_DEFAULT_PID_ID);
        m_rightRearTalon.selectProfileSlot(1, MotorConfig.TALON_DEFAULT_PID_ID);
        m_leftRearTalon.set(ControlMode.Velocity, left, DemandType.ArbitraryFeedForward, MotionControl.DRIVETRAIN_FEEDFORWARD.calculate(left));
        m_rightRearTalon.set(ControlMode.Velocity, right, DemandType.ArbitraryFeedForward, MotionControl.DRIVETRAIN_FEEDFORWARD.calculate(right));
    }

    public void tankDriveVolts(double leftVolts, double rightVolts)
    {
        m_leftRearTalon.setVoltage(leftVolts);
        m_rightRearTalon.setVoltage(rightVolts);
    }

    public Pose2d getPose()
    {
        return m_odometry.getPoseMeters();
    }

    public double getHeading()
    {
        return m_gyro.getRotation2d().getDegrees();
    }

    public DifferentialDriveWheelSpeeds getWheelSpeeds()
    {
        return new DifferentialDriveWheelSpeeds(m_leftRearTalon.getSelectedSensorVelocity() * Units.ENCODER_ANGULAR_VELOCITY.to(Units.METERS_PER_SECOND),
        m_rightRearTalon.getSelectedSensorVelocity() * Units.ENCODER_ANGULAR_VELOCITY.to(Units.METERS_PER_SECOND));
    }

    // Distance (meters) = wheel radius * angle traveled (in radians)
    public double getDistanceLeft() {
        return (RobotMeasurements.DRIVETRAIN_WHEEL_RADIUS_METERS) * (m_leftRearTalon.getSelectedSensorPosition() * Units.ENCODER_ANGLE.to(Units.RADIAN));
    }

    public double getDistanceRight() {
        return (RobotMeasurements.DRIVETRAIN_WHEEL_RADIUS_METERS) * (m_rightRearTalon.getSelectedSensorPosition() * Units.ENCODER_ANGLE.to(Units.RADIAN));
    }

    public void zeroHeading()
    {
        m_gyro.reset();
    }

    public void resetEncoders()
    {
        m_leftRearTalon.setSelectedSensorPosition(0);
        m_rightRearTalon.setSelectedSensorPosition(0);
    }

    public void resetOdometry(Pose2d pose)
    {
        resetEncoders();
        m_odometry.resetPosition(m_gyro.getRotation2d(), this.getDistanceLeft(), this.getDistanceRight(), pose);
    }

    public boolean inPosition()
    {
        return Math.abs(m_rightRearTalon.getClosedLoopError()) < 250;
    }

    public void driveMeters(double meters)
    {
        m_leftRearTalon.selectProfileSlot(0, MotorConfig.TALON_DEFAULT_PID_ID);
        m_rightRearTalon.selectProfileSlot(0, MotorConfig.TALON_DEFAULT_PID_ID);
        double ticks = (meters/(RobotMeasurements.DRIVETRAIN_WHEEL_RADIUS_METERS*2.0*Math.PI))*MotorConfig.TALON_ENCODER_RESOLUTION;
        m_leftRearTalon.set(ControlMode.Position, m_leftRearTalon.getSelectedSensorPosition()+ticks);
        m_rightRearTalon.set(ControlMode.Position, m_rightRearTalon.getSelectedSensorPosition()+ticks);
    }

    private void configureMotors()
    {
        //First setup talons with default settings
        m_leftFrontTalon.configFactoryDefault();
        m_leftRearTalon.configFactoryDefault();
        m_rightFrontTalon.configFactoryDefault();
        m_rightRearTalon.configFactoryDefault();

        // Both encoders inverted on test drivetrain
        m_leftRearTalon.setSensorPhase(true);
        m_rightRearTalon.setSensorPhase(true);

        m_rightRearTalon.setInverted(true);
        m_rightFrontTalon.setInverted(true);

        m_leftFrontTalon.follow(m_leftRearTalon, MotorConfig.DEFAULT_MOTOR_FOLLOWER_TYPE);
        m_rightFrontTalon.follow(m_rightRearTalon, MotorConfig.DEFAULT_MOTOR_FOLLOWER_TYPE);

        //Setup talon built-in PID
        m_leftRearTalon.configSelectedFeedbackSensor(MotorConfig.TALON_DEFAULT_FEEDBACK_DEVICE, MotorConfig.TALON_DEFAULT_PID_ID, MotorConfig.TALON_TIMEOUT_MS);
        m_rightRearTalon.configSelectedFeedbackSensor(MotorConfig.TALON_DEFAULT_FEEDBACK_DEVICE, MotorConfig.TALON_DEFAULT_PID_ID, MotorConfig.TALON_TIMEOUT_MS);
        
        //Create config objects
        TalonSRXConfiguration cLeft = new TalonSRXConfiguration(), cRight = new TalonSRXConfiguration();

        //Setup config objects with desired values
        // Using feedforward, we need to use the PID values given by SysID
        cLeft.slot0 = MotionControl.DRIVETRAIN_LEFT_PID;
        cRight.slot0 = MotionControl.DRIVETRAIN_RIGHT_PID;

        cLeft.slot1 = MotionControl.DRIVETRAIN_LEFT_VELOCITY_PID;
        cRight.slot1 = MotionControl.DRIVETRAIN_RIGHT_VELOCITY_PID;

        NeutralMode mode = NeutralMode.Brake;

        //Brake mode so no coasting
        m_leftRearTalon.setNeutralMode(mode);
        m_leftFrontTalon.setNeutralMode(mode);
        m_rightRearTalon.setNeutralMode(mode);
        m_rightFrontTalon.setNeutralMode(mode);

        //Configure talons
        m_leftRearTalon.configAllSettings(cLeft);
        m_rightRearTalon.configAllSettings(cRight);
    }
}
