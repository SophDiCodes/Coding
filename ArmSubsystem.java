package frc.robot.subsystems;

import com.revrobotics.SparkMaxPIDController;
import com.revrobotics.CANSparkMax;
import com.revrobotics.RelativeEncoder;
import com.revrobotics.CANSparkMax.ControlType; 
import com.revrobotics.CANSparkMaxLowLevel.MotorType;

import edu.wpi.first.wpilibj2.command.InstantCommand;
import edu.wpi.first.wpilibj2.command.Subsystem;
import edu.wpi.first.wpilibj2.command.SubsystemBase;
import frc.robot.Constants.MotionControl;
import edu.wpi.first.wpilibj.DigitalInput;
import edu.wpi.first.wpilibj.Servo;
import edu.wpi.first.wpilibj.shuffleboard.Shuffleboard;
import edu.wpi.first.wpilibj.smartdashboard.SmartDashboard;

import static frc.robot.Constants.*;

public class ArmSubsystem extends SubsystemBase{
    private CANSparkMax armLeft, armRight;
    private SparkMaxPIDController m_pidController;
    private RelativeEncoder m_encoder;
    private DigitalInput m_frontLimit, m_backLimit;
    private Servo m_eyebrow1;
    private Servo m_eyebrow2;


    public ArmSubsystem() {
        //Instantiates two SparkMax motors
        armLeft = new CANSparkMax(Ports.CAN_ARM_LEFT_SPARKMAX, MotorType.kBrushless);
        armRight = new CANSparkMax(Ports.CAN_ARM_RIGHT_SPARKMAX, MotorType.kBrushless);

        //Creates two Limit Switches
        m_frontLimit = new DigitalInput(Ports.DIO_FRONT_LIMIT_SWITCH);
        m_backLimit = new DigitalInput(Ports.DIO_BACK_LIMIT_SWITCH);
        
        m_pidController = armRight.getPIDController();
        m_encoder = armRight.getEncoder();

        m_eyebrow1 = new Servo(0);
        m_eyebrow2 = new Servo(1);

        SmartDashboard.putData("Reset Encoder", new InstantCommand(
            () -> resetArmPosition()
        ));

        //Runs configureMotors
        configureMotors();
    }

    private void configureMotors() {
        //Set motor to default 
        armLeft.restoreFactoryDefaults();
        armRight.restoreFactoryDefaults();
        
        //PID
        m_pidController.setP(MotionControl.ARM_PID.kP);
        m_pidController.setI(MotionControl.ARM_PID.kI);
        m_pidController.setD(MotionControl.ARM_PID.kD);

        armLeft.setClosedLoopRampRate(MotionControl.CLOSED_LOOP_RAMP_RATE);
        armRight.setClosedLoopRampRate(MotionControl.CLOSED_LOOP_RAMP_RATE);
        armLeft.setOpenLoopRampRate(MotionControl.OPEN_LOOP_RAMP_RATE);
        armRight.setOpenLoopRampRate(MotionControl.OPEN_LOOP_RAMP_RATE);

        //Left follows right
        armLeft.follow(armRight, true);

        armLeft.setSmartCurrentLimit(22, 25);
        armRight.setSmartCurrentLimit(22, 25);
    }

    public boolean isArmAtLimit() {
       // checks if either of the limit switch is triggered 
        return (m_frontLimit.get() || m_backLimit.get());
    }

    //Takes in number of rotations for arm (without gear reduction)
    public void moveToPosition(double setPoint) {
        m_pidController.setReference(setPoint, CANSparkMax.ControlType.kPosition);
    }

    public void resetArmPosition() {
        m_encoder.setPosition(0);
    }

    public void armPercentOutput(double percent_output) {
        armRight.set(percent_output);
    }

    public double getPosition() {
        return (m_encoder.getPosition());
    }

    public void eyebrowPose(double position)
    {
        m_eyebrow1.set(position);
        m_eyebrow2.set(1-position);
    }

    @Override
    public void periodic() {
        
    }
}
