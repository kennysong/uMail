class CreateExperiments < ActiveRecord::Migration
  def self.up
    create_table :experiments do |t|
      t.column :id, :primary_key
      t.column :Admin, :string
      t.column :participant_id, :integer
      t.column :Location, :string
      t.column :Date, :datetime
    end
  end

  def self.down
    drop_table :experiments
  end
end
