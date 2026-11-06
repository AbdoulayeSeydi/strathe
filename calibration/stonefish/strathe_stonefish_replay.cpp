// strathe_stonefish_replay: replays logged windows of a vehicle in Stonefish, headless, for `strathe fit-stonefish`
// (PROJECT_STATUS.md #72). Each window starts from a logged state (velocity at the logs' reference point, attitude) and
// is driven by the logged body wrench, applied every physics tick; Stonefish integrates the motion and the attitude
// with the vehicle's own model (mass, buoyancy, drag, added mass).
//
//   strathe_stonefish_replay --scenario S.scn --data DIR --robot NAME --ref X Y Z [--depth 3] [--hz 200] [--dt 0.1] --info
//   strathe_stonefish_replay ... --windows IN.csv --out OUT.csv
//
// IN.csv rows:  win,k,u,v,w,p,q,r,qx,qy,qz,qw,Fx,Fy,Fz,Tx,Ty,Tz  (k = 0, 1, ... per window, one log sample apart)
//   velocity: body FRD at the reference point; quaternion: body->world (NED); wrench: body FRD about the reference point
// OUT.csv rows: win,k,u,v,w,p,q,r,qx,qy,qz,qw  (Stonefish's prediction; k = 0 is the logged start)
// --info prints the water density, the robot's base link and its parts (mass, drag coefficients, added mass, volume,
// and the mesh areas Stonefish's drag uses) as JSON.
// Frames: Stonefish's body frame is forward-right-down and its world NED, the same as Strathe's.
#include <core/ConsoleSimulationApp.h>
#include <core/SimulationManager.h>
#include <core/ScenarioParser.h>
#include <core/FeatherstoneRobot.h>
#include <entities/SolidEntity.h>
#include <entities/FeatherstoneEntity.h>
#include <entities/solids/Compound.h>
#include <entities/forcefields/Ocean.h>
#include <graphics/OpenGLDataStructs.h>
#include <actuators/Actuator.h>
#include <actuators/Rudder.h>
#include <BulletDynamics/Featherstone/btMultiBody.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace sf;

#ifdef STONEFISH_ADDED_MASS_PATCH
static const bool kPatched = true;
#else
static const bool kPatched = false;
#endif
#ifdef STONEFISH_AXIS_DAMPING_PATCH
static const bool kAxisDamping = true;  // the patch's per-axis damping (damping_linear etc. in <hydrodynamics>)
#else
static const bool kAxisDamping = false;
#endif

class ReplayManager : public SimulationManager {
public:
    ReplayManager(std::string scn, Scalar hz) : SimulationManager(hz), scn_(std::move(scn)) {}
    void BuildScenario() override {
        ScenarioParser p(this);
        if (!p.Parse(scn_)) { fprintf(stderr, "can't parse the scenario %s\n", scn_.c_str()); exit(3); }
    }
private:
    std::string scn_;
};

class ReplayApp : public ConsoleSimulationApp {  // Init() is protected: build the scenario without the real-time loop
public:
    using ConsoleSimulationApp::ConsoleSimulationApp;
    void Build() { Init(); }
};

// Applies the logged wrench every tick: Stonefish clears all forces at the start of a tick, then updates actuators.
class WrenchInjector : public Actuator {
public:
    WrenchInjector() : Actuator("strathe_logged_wrench") {}
    SolidEntity* body = nullptr;
    Vector3 ref = V0(), F_body = V0(), T_ref_body = V0();
    void Update(Scalar dt) override {
        Actuator::Update(dt);
        if (!body) return;
        Transform o = body->getOTransform(), cg = body->getCGTransform();
        Vector3 r_cg = o.getBasis().transpose() * (cg.getOrigin() - o.getOrigin());
        body->ApplyCentralForce(o.getBasis() * F_body);
        body->ApplyTorque(o.getBasis() * (T_ref_body + (ref - r_cg).cross(F_body)));
    }
    ActuatorType getType() const override { return ActuatorType::PUSH; }
};

static void vec(FILE* f, const Vector3& v) { fprintf(f, "[%.9g, %.9g, %.9g]", v.x(), v.y(), v.z()); }

// The sums Stonefish's drag uses, over the part's physics mesh in the part's own (O) axes, for a translation along each
// axis: quadratic drag = 0.5 rho Cd * sum(A * max(n_i, 0)) * |v|^2 * |v_n| (the projected area), viscous drag = rho Cf *
// sum(A * (1 - n_i^2)) * v (the area the flow slides along). `strathe export-stonefish` converts Strathe's drag with them.
static void drag_areas(FILE* f, SolidEntity* s) {
    Vector3 proj = V0(), tang = V0();
    const Mesh* m = s->getPhysicsMesh();
    if (m) {
        Transform t = s->getO2CTransform();
        for (size_t i = 0; i < m->faces.size(); ++i) {
            Vector3 p[3];
            for (unsigned short j = 0; j < 3; ++j) {
                glm::vec3 g = m->getVertexPos(i, j);
                p[j] = t * Vector3(g.x, g.y, g.z);
            }
            Vector3 n = (p[1] - p[0]).cross(p[2] - p[0]);
            Scalar len = n.length();
            if (len < Scalar(1e-12)) continue;
            n /= len;
            Scalar A = len / 2;
            for (int k = 0; k < 3; ++k) {
                proj[k] += A * btMax(n[k], Scalar(0));
                tang[k] += A * (1 - n[k] * n[k]);
            }
        }
    }
    fprintf(f, ", \"projected_area\": ");
    vec(f, proj);
    fprintf(f, ", \"tangential_area\": ");
    vec(f, tang);
}

static void info(FILE* f, SimulationManager* sim, FeatherstoneRobot* robot, SolidEntity* body) {
    Vector3 Cd, Cf;
    Ocean* ocn = sim->getOcean();
    fprintf(f, "{\"patched\": %s, \"axis_damping\": %s, \"water_density\": %.9g, \"base\": {\"name\": \"%s\", "
               "\"mass\": %.9g, \"augmented_mass\": %.9g, \"volume\": %.9g, \"inertia\": ", kPatched ? "true" : "false",
            kAxisDamping ? "true" : "false",
            ocn ? ocn->getLiquid().density : 0.0, body->getName().c_str(), body->getMass(), body->getAugmentedMass(),
            body->getVolume());
    vec(f, body->getAugmentedInertia());
    fprintf(f, ", \"added_mass\": ");
    vec(f, body->getAddedMass());
    Transform cg2o = body->getCG2OTransform().inverse();  // CG frame -> body (O) frame
    fprintf(f, ", \"cg\": ");
    vec(f, cg2o.getOrigin());
    fprintf(f, ", \"cb\": ");
    vec(f, cg2o * body->getCB());
    Matrix3 R = cg2o.getBasis();  // the inertia (with added inertia) along the body's own axes, about the CG
    Vector3 Ip = body->getAugmentedInertia();
    Matrix3 Io = R * Matrix3(Ip.x(), 0, 0, 0, Ip.y(), 0, 0, 0, Ip.z()) * R.transpose();
    fprintf(f, ", \"inertia_body_axes\": ");
    vec(f, Vector3(Io[0][0], Io[1][1], Io[2][2]));
#ifdef STONEFISH_ADDED_MASS_PATCH
    fprintf(f, ", \"anisotropic_added_mass\": ");
    vec(f, body->getAnisotropicAddedMass());
#endif
    fprintf(f, "}, \"parts\": [");
    if (body->getSolidType() == SolidType::COMPOUND) {
        Compound* c = (Compound*)body;
        // Compound::getPart returns a part with no solid past the last one, in unpatched Stonefish too (#76: the
        // part count used to come from the patch, so unpatched Stonefish listed none and both commands failed)
        for (size_t i = 0; c->getPart(i).solid != nullptr; ++i) {
            SolidEntity* s = c->getPart(i).solid;
            s->getHydrodynamicCoefficients(Cd, Cf);
            fprintf(f, "%s{\"name\": \"%s\", \"mass\": %.9g, \"volume\": %.9g, \"buoyant\": %s, \"external\": %s, "
                       "\"quadratic_drag\": ", i ? ", " : "", s->getName().c_str(), s->getMass(), s->getVolume(),
                    s->isBuoyant() ? "true" : "false", c->getPart(i).isExternal ? "true" : "false");
            vec(f, Cd);
            fprintf(f, ", \"viscous_drag\": ");
            vec(f, Cf);
            fprintf(f, ", \"added_mass\": ");
            vec(f, s->getAddedMass());
            fprintf(f, ", \"added_inertia\": ");
            vec(f, s->getAddedInertia());
            fprintf(f, ", \"inertia\": ");
            vec(f, s->getInertia());
            drag_areas(f, s);
            fprintf(f, "}");
        }
    }
    fprintf(f, "]}\n");
}

int main(int argc, char** argv) {
    std::string scn, data, robotName, windows, out;
    Vector3 ref = V0();
    double depth = 3.0, hz = 200.0, sampleDt = 0.1;
    bool doInfo = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](int k = 1) { if (i + k >= argc) { fprintf(stderr, "%s needs a value\n", a.c_str()); exit(2); } };
        if (a == "--scenario") { next(); scn = argv[++i]; }
        else if (a == "--data") { next(); data = argv[++i]; }
        else if (a == "--robot") { next(); robotName = argv[++i]; }
        else if (a == "--ref") { next(3); ref = Vector3(atof(argv[i + 1]), atof(argv[i + 2]), atof(argv[i + 3])); i += 3; }
        else if (a == "--depth") { next(); depth = atof(argv[++i]); }
        else if (a == "--hz") { next(); hz = atof(argv[++i]); }
        else if (a == "--dt") { next(); sampleDt = atof(argv[++i]); }
        else if (a == "--windows") { next(); windows = argv[++i]; }
        else if (a == "--out") { next(); out = argv[++i]; }
        else if (a == "--info") doInfo = true;
        else if (a == "--features") {  // what this build supports, without a scenario (`strathe check`)
            printf("{\"patched\": %s, \"axis_damping\": %s, \"fins\": true}\n", kPatched ? "true" : "false",
                   kAxisDamping ? "true" : "false");
            return 0;
        }
        else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (scn.empty() || data.empty() || robotName.empty() || (!doInfo && (windows.empty() || out.empty()))) {
        fprintf(stderr, "usage: strathe_stonefish_replay --scenario S --data DIR --robot NAME --ref X Y Z "
                        "[--depth D] [--hz HZ] [--dt LOG_DT] (--info | --windows IN.csv --out OUT.csv)\n");
        return 2;
    }
    if (!data.empty() && data.back() != '/') data += "/";

    std::map<int, std::vector<std::vector<double>>> wins;
    std::vector<std::string> finNames;   // a fin-steered vehicle's windows name each fin after the wrench: fin_<name>, its rudder actuator
    if (!doInfo) {
        std::ifstream in(windows);
        if (!in) { fprintf(stderr, "can't read %s\n", windows.c_str()); return 2; }
        std::string line;
        while (std::getline(in, line)) {
            if (!line.empty() && line[0] == 'w') {
                std::stringstream hs(line);
                std::string cell;
                int col = 0;
                while (std::getline(hs, cell, ',')) if (col++ >= 18) finNames.push_back(cell);
                continue;
            }
            if (line.empty()) continue;
            std::stringstream ss(line);
            std::string tok;
            std::vector<double> v;
            while (std::getline(ss, tok, ',')) v.push_back(std::stod(tok));
            if (v.size() != 18 + finNames.size()) { fprintf(stderr, "bad window row (%zu values, need %zu)\n", v.size(), 18 + finNames.size()); return 2; }
            wins[(int)v[0]].push_back(std::vector<double>(v.begin() + 2, v.end()));
        }
    }

    ReplayManager sim(scn, hz);
    ReplayApp app("strathe_stonefish_replay", data, &sim);
    app.Build();
    Robot* r = sim.getRobot(robotName);
    if (!r) { fprintf(stderr, "no robot named '%s' in the scenario\n", robotName.c_str()); return 3; }
    FeatherstoneRobot* robot = dynamic_cast<FeatherstoneRobot*>(r);
    if (!robot) { fprintf(stderr, "robot '%s' isn't a multibody robot\n", robotName.c_str()); return 3; }
    SolidEntity* body = robot->getBaseLink();
    if (doInfo) { info(stdout, &sim, robot, body); return 0; }

    std::vector<Rudder*> fins;
    for (const auto& name : finNames) {
        Actuator* act = robot->getActuator(robotName + "/" + name);
        Rudder* rd = act == nullptr ? nullptr : dynamic_cast<Rudder*>(act);
        if (rd == nullptr) { fprintf(stderr, "the robot has no rudder actuator named %s: each fin needs one (FINS.md)\n", name.c_str()); return 2; }
        fins.push_back(rd);
    }
    btMultiBody* mb = robot->getDynamics()->getMultiBody();
    WrenchInjector* inj = new WrenchInjector();
    inj->body = body;
    inj->ref = ref;
    sim.AddActuator(inj);
    if (!sim.StartSimulation()) { fprintf(stderr, "Stonefish couldn't start the simulation\n"); return 3; }

    FILE* fo = fopen(out.c_str(), "w");
    if (!fo) { fprintf(stderr, "can't write %s\n", out.c_str()); return 2; }
    fprintf(fo, "win,k,u,v,w,p,q,r,qx,qy,qz,qw\n");
    const Scalar h = Scalar(1) / hz;
    for (auto& [win, rows] : wins) {
        const std::vector<double>& r0 = rows[0];
        Matrix3 R(Quaternion(r0[6], r0[7], r0[8], r0[9]));
        Transform T_wO(R, Vector3(0, 0, depth));
        Transform T_wCG = T_wO * body->getCG2OTransform().inverse();
        robot->getDynamics()->setBaseTransform(T_wCG * body->getCG2CTransform());
        Vector3 w_world = R * Vector3(r0[3], r0[4], r0[5]);
        mb->setBaseVel(R * Vector3(r0[0], r0[1], r0[2]) + w_world.cross(T_wCG.getOrigin() - T_wO * ref));
        mb->setBaseOmega(w_world);
#ifdef STONEFISH_ADDED_MASS_PATCH
        body->ResetAddedMassState();
#endif
        for (size_t k = 0; k < rows.size(); ++k) {
            Transform o = body->getOTransform(), cg = body->getCGTransform();
            Matrix3 Ro = o.getBasis();
            Vector3 w = body->getAngularVelocity();
            Vector3 vp = body->getLinearVelocity() + w.cross(o * ref - cg.getOrigin());
            Vector3 vb = Ro.transpose() * vp, wb = Ro.transpose() * w;
            Quaternion qo = o.getRotation();
            fprintf(fo, "%d,%zu,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n", win, k, vb.x(), vb.y(), vb.z(),
                    wb.x(), wb.y(), wb.z(), qo.x(), qo.y(), qo.z(), qo.w());
            if (k + 1 == rows.size()) break;
            inj->F_body = Vector3(rows[k][10], rows[k][11], rows[k][12]);
            inj->T_ref_body = Vector3(rows[k][13], rows[k][14], rows[k][15]);
            for (size_t i = 0; i < fins.size(); ++i) fins[i]->setSetpoint(rows[k][16 + i]);
            int sub = (int)std::lround(sampleDt * hz);
            for (int i = 0; i < sub; ++i) sim.StepSimulation(h);
            if (!std::isfinite(body->getLinearVelocity().x())) { fprintf(stderr, "window %d diverged\n", win); break; }
        }
    }
    fclose(fo);
    return 0;
}
