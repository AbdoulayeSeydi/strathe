// strathe_gz_replay: replay logged windows in Gazebo (gz-sim 8, Harmonic) for `strathe fit-sim --sim gazebo`
// (PROJECT_STATUS.md #82). It speaks Strathe's replay contract (calibration/simulators/SIMULATOR_PROTOCOL.md):
//
//   strathe_gz_replay --model WORLD.sdf --ref X Y Z --dt DT [--model-name NAME] [--link LINK] [--depth D]
//                     (--features | --info | --windows IN.csv --out OUT.csv)
//
// The world is loaded headless with its own systems (Physics, Buoyancy, Hydrodynamics...). For each window the
// vehicle is put at --depth with the logged attitude and velocity, then driven by the logged wrench, applied at the
// reference point every physics step, and left to move on its own. Strathe's frames are north-east-down / forward-
// right-down; Gazebo's east-north-up / forward-left-up: converted here. The reference point is relative to the link's
// origin, in the link's axes as forward-right-down. Thrusters and controllers in the world aren't commanded.
#include <gz/math/Matrix3.hh>
#include <gz/math/Pose3.hh>
#include <gz/math/Quaternion.hh>
#include <gz/math/Vector3.hh>
#include <gz/sim/EntityComponentManager.hh>
#include <gz/sim/Joint.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/Server.hh>
#include <gz/sim/ServerConfig.hh>
#include <gz/sim/TestFixture.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/World.hh>
#include <gz/sim/components/AngularVelocityCmd.hh>
#include <gz/sim/components/Collision.hh>
#include <gz/sim/components/Geometry.hh>
#include <gz/sim/components/Inertial.hh>
#include <gz/sim/components/LinearVelocityCmd.hh>
#include <gz/sim/components/Model.hh>
#include <gz/sim/components/Name.hh>
#include <gz/sim/components/ParentEntity.hh>
#include <gz/sim/components/Pose.hh>
#include <gz/sim/components/Static.hh>
#include <gz/sim/config.hh>
#include <sdf/Box.hh>
#include <sdf/Cylinder.hh>
#include <sdf/Geometry.hh>
#include <sdf/Root.hh>
#include <sdf/Sphere.hh>
#include <sdf/World.hh>
#include <sdf/Physics.hh>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace gz;

static const math::Matrix3d T_NED_ENU(0, 1, 0, 1, 0, 0, 0, 0, -1);  // world: NED <-> ENU (its own inverse)
static const math::Matrix3d T_FRD_FLU(1, 0, 0, 0, -1, 0, 0, 0, -1);  // body: FRD <-> FLU (its own inverse)

static math::Vector3d B(const math::Vector3d &v) { return T_FRD_FLU * v; }  // body FRD <-> FLU

static void fail(const std::string &m) { fprintf(stderr, "%s\n", m.c_str()); exit(2); }

struct Args {
  std::string world, modelName, linkName, in, out;
  math::Vector3d ref;
  double dt = 0.1, depth = 3.0;
  bool info = false;
};

static Args parse(int argc, char **argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    auto need = [&](int n) { if (i + n >= argc) fail(s + " needs a value"); };
    if (s == "--model") { need(1); a.world = argv[++i]; }
    else if (s == "--model-dir") { need(1); ++i; }
    else if (s == "--ref") { need(3); a.ref.Set(atof(argv[i + 1]), atof(argv[i + 2]), atof(argv[i + 3])); i += 3; }
    else if (s == "--dt") { need(1); a.dt = atof(argv[++i]); }
    else if (s == "--depth") { need(1); a.depth = atof(argv[++i]); }
    else if (s == "--model-name") { need(1); a.modelName = argv[++i]; }
    else if (s == "--link") { need(1); a.linkName = argv[++i]; }
    else if (s == "--windows") { need(1); a.in = argv[++i]; }
    else if (s == "--out") { need(1); a.out = argv[++i]; }
    else if (s == "--info") a.info = true;
    else fail("unknown argument " + s);
  }
  if (a.world.empty()) fail("--model WORLD.sdf is required");
  if (!a.info && (a.in.empty() || a.out.empty())) fail("give --info, or --windows IN.csv --out OUT.csv");
  return a;
}

// Volume and centre (in the link frame) of a primitive collision; volume 0 for meshes and others.
static void primitive(const sdf::Geometry &g, double &vol, std::string &type) {
  vol = 0; type = "other";
  if (g.Type() == sdf::GeometryType::BOX) { auto s = g.BoxShape()->Size(); vol = s.X() * s.Y() * s.Z(); type = "box"; }
  else if (g.Type() == sdf::GeometryType::CYLINDER) {
    auto c = g.CylinderShape(); vol = GZ_PI * c->Radius() * c->Radius() * c->Length(); type = "cylinder"; }
  else if (g.Type() == sdf::GeometryType::SPHERE) {
    double r = g.SphereShape()->Radius(); vol = 4.0 / 3.0 * GZ_PI * r * r * r; type = "sphere"; }
  else if (g.Type() == sdf::GeometryType::MESH) type = "mesh";
}

int main(int argc, char **argv) {
  for (int i = 1; i < argc; ++i)
    if (std::string(argv[i]) == "--features") {
      printf("{\"simulator\": \"gazebo\", \"gz_sim\": \"%s\", \"fins\": true}\n", GZ_SIM_VERSION_FULL);
      return 0;
    }
  Args a = parse(argc, argv);

  sdf::Root root;
  auto errs = root.Load(a.world);
  if (!errs.empty()) fail("can't load " + a.world + ": " + errs[0].Message());
  if (root.WorldCount() < 1) fail(a.world + " has no <world>");
  double step = root.WorldByIndex(0)->PhysicsDefault()->MaxStepSize();
  int sub = std::max(1, (int)std::lround(a.dt / step));
  if (std::fabs(sub * step - a.dt) > 1e-9 * std::max(1.0, a.dt))
    fail("the logs' sample time (" + std::to_string(a.dt) + " s) isn't a whole number of the world's physics steps (" +
         std::to_string(step) + " s): set <max_step_size> so it divides it");

  // windows: win -> rows of 16 numbers (u v w p q r qx qy qz qw Fx Fy Fz Tx Ty Tz)
  std::vector<std::string> finNames;
  std::vector<sim::Entity> finJoints;
  std::map<int, std::vector<std::vector<double>>> wins;
  if (!a.info) {
    std::ifstream in(a.in);
    if (!in) fail("can't read " + a.in);
    std::string line;
    std::getline(in, line);
    {  // a fin-steered vehicle's windows name each fin after the wrench: fin_<name>, the joint that turns it
      std::stringstream hs(line);
      std::string cell;
      int col = 0;
      while (std::getline(hs, cell, ',')) {
        while (!cell.empty() && (cell.back() == '\r' || cell.back() == ' ')) cell.pop_back();
        if (col++ >= 18) finNames.push_back(cell);
      }
    }
    while (std::getline(in, line)) {
      std::stringstream ss(line);
      std::string cell;
      std::vector<double> v;
      while (std::getline(ss, cell, ',')) v.push_back(atof(cell.c_str()));
      if (v.size() != 18 + finNames.size()) fail("a row of " + a.in + " doesn't have the header's columns");
      wins[(int)v[0]].push_back(std::vector<double>(v.begin() + 2, v.end()));
    }
  }

  if (a.info) {
    // what the vehicle's link has: mass, inertia, centre of mass and collision volumes (buoyancy uses them)
    sdf::Root r2;
    r2.Load(a.world);
    const sdf::World *w = r2.WorldByIndex(0);
    const sdf::Model *m = nullptr;
    for (uint64_t i = 0; i < w->ModelCount(); ++i)
      if ((a.modelName.empty() && !w->ModelByIndex(i)->Static()) || w->ModelByIndex(i)->Name() == a.modelName) {
        m = w->ModelByIndex(i);
        if (!a.modelName.empty() || !m->Static()) break;
      }
    if (!m) fail("no model in " + a.world);
    const sdf::Link *l = a.linkName.empty() ? m->LinkByIndex(0) : m->LinkByName(a.linkName);
    if (!a.linkName.empty() || l == nullptr) l = a.linkName.empty() ? m->LinkByIndex(0) : l;
    if (!l) fail("no link");
    auto in = l->Inertial();
    auto com = in.Pose().Pos();
    auto M = in.MassMatrix().DiagonalMoments();
    printf("{\"simulator\": \"gazebo\", \"gz_sim\": \"%s\", \"model\": \"%s\", \"link\": \"%s\", \"mass\": %.9g, "
           "\"inertia\": [%.9g, %.9g, %.9g], \"com_flu\": [%.9g, %.9g, %.9g], \"max_step_size\": %.9g, \"collisions\": [",
           GZ_SIM_VERSION_FULL, m->Name().c_str(), l->Name().c_str(), in.MassMatrix().Mass(), M.X(), M.Y(), M.Z(),
           com.X(), com.Y(), com.Z(), step);
    for (uint64_t i = 0; i < l->CollisionCount(); ++i) {
      const sdf::Collision *c = l->CollisionByIndex(i);
      double vol; std::string type;
      primitive(*c->Geom(), vol, type);
      gz::math::Pose3d cp;
      c->SemanticPose().Resolve(cp, l->Name());
      printf("%s{\"name\": \"%s\", \"type\": \"%s\", \"volume\": %.9g, \"center_flu\": [%.9g, %.9g, %.9g]}",
             i ? ", " : "", c->Name().c_str(), type.c_str(), vol, cp.Pos().X(), cp.Pos().Y(), cp.Pos().Z());
    }
    printf("]}\n");
    return 0;
  }

  sim::ServerConfig cfg;
  cfg.SetSdfFile(a.world);
  sim::TestFixture fix(cfg);
  sim::Entity modelE = sim::kNullEntity, linkE = sim::kNullEntity;
  math::Pose3d linkInModel;
  std::string err;

  // state of the replay
  // RESET: put the vehicle at the logged pose, at rest; SPEED: give it the logged velocity (in its own axes, which
  // need the new attitude to be in place: one step after the pose); RUN: the logged wrench, every step.
  enum Phase { RESET, SPEED, RUN } phase = RESET;
  std::vector<std::vector<double>> *rows = nullptr;  // the current window
  size_t k = 0;
  int s = 0;
  bool diverged = false;
  std::vector<std::vector<double>> got;  // recorded samples of the current window
  math::Vector3d refFLU = B(a.ref);

  fix.OnConfigure([&](const sim::Entity &, const std::shared_ptr<const sdf::Element> &,
                      sim::EntityComponentManager &ecm, sim::EventManager &) {
    ecm.Each<sim::components::Model, sim::components::Name>(
        [&](const sim::Entity &e, const sim::components::Model *, const sim::components::Name *n) {
          auto st = ecm.Component<sim::components::Static>(e);
          if ((a.modelName.empty() && !(st && st->Data()) && modelE == sim::kNullEntity) || n->Data() == a.modelName)
            modelE = e;
          return true;
        });
    if (modelE == sim::kNullEntity) { err = "no model " + (a.modelName.empty() ? std::string("that moves") : a.modelName); return; }
    sim::Model m(modelE);
    linkE = a.linkName.empty() ? m.CanonicalLink(ecm) : m.LinkByName(ecm, a.linkName);
    if (linkE == sim::kNullEntity) { err = "model has no link " + a.linkName; return; }
    for (const auto &name : finNames) {
      sim::Entity j = m.JointByName(ecm, name);
      if (j == sim::kNullEntity) { err = "model has no joint " + name + ": each fin needs one (FINS.md)"; return; }
      finJoints.push_back(j);
    }
    sim::Link(linkE).EnableVelocityChecks(ecm, true);
    auto p = ecm.Component<sim::components::Pose>(linkE);
    if (p) linkInModel = p->Data();
  });

  fix.OnPreUpdate([&](const sim::UpdateInfo &, sim::EntityComponentManager &ecm) {
    if (!rows || linkE == sim::kNullEntity) return;
    sim::Link link(linkE);
    const auto &r = (*rows)[std::min(k, rows->size() - 1)];
    for (size_t i = 0; i < finJoints.size(); ++i) {   // held at the logged angle: LiftDrag makes the force from the real flow
      sim::Joint j(finJoints[i]);
      j.ResetPosition(ecm, {r[16 + i]});
      j.ResetVelocity(ecm, {0.0});
    }
    if (phase == RESET) {
      // the logged attitude (body FRD -> NED) as Gazebo's (link FLU -> ENU), at --depth
      math::Quaterniond qn(r[9], r[6], r[7], r[8]);  // w, x, y, z
      math::Matrix3d Rn(qn);
      math::Matrix3d Re = T_NED_ENU * Rn * T_FRD_FLU;
      math::Pose3d linkWorld(math::Vector3d(0, 0, -a.depth), math::Quaterniond(Re));
      sim::Model(modelE).SetWorldPoseCmd(ecm, linkWorld * linkInModel.Inverse());
      link.SetLinearVelocity(ecm, math::Vector3d::Zero);
      link.SetAngularVelocity(ecm, math::Vector3d::Zero);
      return;
    }
    if (phase == SPEED) {
      math::Vector3d w(r[3], r[4], r[5]), vref(r[0], r[1], r[2]);
      math::Vector3d vo = vref - w.Cross(a.ref);  // the link origin's velocity, body FRD
      link.SetLinearVelocity(ecm, B(vo));
      link.SetAngularVelocity(ecm, B(w));
      return;
    }
    ecm.RemoveComponent<sim::components::LinearVelocityCmd>(linkE);
    ecm.RemoveComponent<sim::components::AngularVelocityCmd>(linkE);
    auto pose = link.WorldPose(ecm);
    if (!pose) return;
    math::Matrix3d R(pose->Rot());
    math::Vector3d F(r[10], r[11], r[12]), Tq(r[13], r[14], r[15]);
    link.AddWorldWrench(ecm, R * B(F), R * B(Tq), refFLU);
  });

  auto sample = [&](const sim::EntityComponentManager &ecm) {
    sim::Link link(linkE);
    auto pose = link.WorldPose(ecm);
    auto v = link.WorldLinearVelocity(ecm, refFLU);
    auto w = link.WorldAngularVelocity(ecm);
    std::vector<double> o(10, NAN);
    if (!pose || !v || !w) return o;
    math::Matrix3d R(pose->Rot());
    math::Vector3d vb = B(R.Transposed() * *v), wb = B(R.Transposed() * *w);
    math::Quaterniond qn(math::Matrix3d(T_NED_ENU * R * T_FRD_FLU));
    o = {vb.X(), vb.Y(), vb.Z(), wb.X(), wb.Y(), wb.Z(), qn.X(), qn.Y(), qn.Z(), qn.W()};
    return o;
  };

  fix.OnPostUpdate([&](const sim::UpdateInfo &, const sim::EntityComponentManager &ecm) {
    if (!rows || diverged) return;
    if (phase == RESET) { phase = SPEED; return; }
    if (phase == SPEED) { phase = RUN; s = 0; k = 0; return; }
    if (++s < sub) return;
    s = 0;
    ++k;
    auto o = sample(ecm);
    bool ok = true;
    for (int i = 0; i < 6; ++i) ok = ok && std::isfinite(o[i]) && std::fabs(o[i]) < 1e3;
    if (!ok) { diverged = true; return; }
    got.push_back(o);
  }).Finalize();

  // load the world (one iteration runs OnConfigure; every window then resets the vehicle)
  fix.Server()->Run(true, 1, false);
  if (!err.empty()) fail(err + " in " + a.world + " (give --model-name / --link)");

  FILE *f = fopen(a.out.c_str(), "w");
  if (!f) fail("can't write " + a.out);
  fprintf(f, "win,k,u,v,w,p,q,r,qx,qy,qz,qw\n");
  for (auto &kv : wins) {
    rows = &kv.second;
    phase = RESET; k = 0; s = 0; diverged = false; got.clear();
    const auto &r0 = (*rows)[0];
    fprintf(f, "%d,0", kv.first);
    for (int i = 0; i < 10; ++i) fprintf(f, ",%.9g", r0[i]);
    fprintf(f, "\n");
    size_t n = rows->size();
    fix.Server()->Run(true, 2 + (n - 1) * sub, false);
    for (size_t j = 0; j < got.size() && j + 1 < n; ++j) {
      fprintf(f, "%d,%zu", kv.first, j + 1);
      for (int i = 0; i < 10; ++i) fprintf(f, ",%.9g", got[j][i]);
      fprintf(f, "\n");
    }
  }
  fclose(f);
  return 0;
}
